from google.cloud import texttospeech
import google.generativeai as palm
import openai
import replicate
import boto3
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import requests

import shutil
import re
import base64
import asyncio
import aiohttp
from time import time, sleep 
import os
import logging
import json
from datetime import datetime

################# Config #################
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(processName)s] [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info("Starting Script")
start_time = time()


logging.info("Loading in Environment Variables")

openai_api_key = os.environ.get("OPENAI_API_KEY") # Open AI API
replicate_token = os.getenv("REPLICATE_API_TOKEN") # Replicate API 
palm_api_key = os.environ.get("PALM_API_KEY") # Palm API
google_access_token = os.environ.get("GOOGLE_ACCESS_TOKEN")
google_project = os.environ.get("GOOGLE_PROJECT")

logging.info("Finished Loading Environment Variables.")


logging.info("Configuring Palm Package")
palm.configure(api_key=palm_api_key)
defaults = {
  'model': 'models/chat-bison-001',
  'temperature': 0.25,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
}
logging.info("Done Configuring Palm Package")

# TTS Model
# logging.info("Configure tts package")
# speech_client = texttospeech.TextToSpeechClient()
# logging.info("Done Configuring tts package")

# S3
logging.info("Connecting to s3")
s3_client = boto3.client('s3')
logging.info("Done Connecting to S3")


################# Core Functions #################
content_bucket = "toast-daily-content"
def get_content(articles:list) -> list:
    article_content_list = []

    for article in articles:
        article_id = article["articleID"]
        topic = article["topic"]
        print("Article ID ", article_id)
        logging.info("Loading in article")
        content_s3_file = f"retrieval/{article_id}/article-content.json"
        content_lambda_file = f"/tmp/article-content.json" 
        s3_client.download_file(content_bucket, content_s3_file, content_lambda_file)
        
        with open("/tmp/article-content.json", "r") as file:
            article_content = json.load(file)

        article_content_list.append(article_content)

    return article_content_list



def split_content(content, max_length):
    return [content[i:i+max_length] for i in range(0, len(content), max_length)]

def count_words(string):
    return len(string.split())


async def replicate_summarization(content: str, length: int, context: str, prompt: str, max_retries: int = 3):
    backoff_factor = 1.05
    token_buffer = 100
    model = "mistralai/mixtral-8x7b-instruct-v0.1"

    # Adjusting max_tokens calculation if needed
    max_tokens = count_words(context + prompt + content) + length + token_buffer

    # Prepare the request payload
    payload = {
        "input": {
            "prompt": f"{context + prompt + content}",
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 0.6,
            "max_new_tokens": max_tokens,
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
    }

    # Headers with authorization token
    headers = {
        "Authorization": f"Token {replicate_token}",
        "Content-Type": "application/json"
    }

    # Replicate API endpoint
    replicate_api_url = f"https://api.replicate.com/v1/models/{model}/predictions"

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(replicate_api_url, headers=headers, json=payload) as response:
                    if response.status <=299:
                        response_data = await response.json()
                        # Adjust according to the actual response structure
                        return response_data.get('result', {}).get('text', '')
                    else:
                        raise Exception(f"HTTP Error: {response.status}")
        except Exception as e:
            logging.warning(f"Replicate error occurred: {e}, attempt {attempt + 1} of {max_retries}")
            await asyncio.sleep(int(backoff_factor ** attempt))

    return False



async def openai_summarization(content: str, length: int, context: str, prompt: str, summarization_style: str = "Business Casual", max_retries: int = 3):
    backoff_factor = 1.5
    token_buffer = 100

    max_tokens = count_words(context) + count_words(content) + count_words(prompt) + length + token_buffer
    if max_tokens > 16000:
        max_tokens = 16000

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ],
        "temperature": 1,
        "max_tokens": max_tokens,
        "top_p": 0.58,
        "frequency_penalty": 0.18,
        "presence_penalty": 0
    }

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=json.dumps(data)) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result['choices'][0]['message']['content']
        except Exception as e:
            logging.warning(f"OpenAIError occurred: {e}, attempt {attempt + 1} of {max_retries}")
            await asyncio.sleep(int(backoff_factor ** attempt))

    return False


def modify_palm_string(input_string):
    # Step 1: Remove 'Sure' if it's the first word
    if input_string.startswith("Sure"):
        input_string = input_string[4:].strip()

    # Step 2: Remove the sentence containing the first 'here'
    # Splitting the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', input_string)

    # Finding the sentence with 'here' and removing it
    for i, sentence in enumerate(sentences):
        if 'here' in sentence and i <= 2:
            del sentences[i]

    # Joining the remaining sentences back into a string
    modified_string = ' '.join(sentences)

    modified_string = modified_string.lstrip(". ")

    return modified_string

def palm_summarization(content:str, length:int, context:str, prompt:str, summarization_style:str="Business Casual"):
    messages = []
    messages.append(prompt)
    messages.append("NEXT REQUEST")
    response = palm.chat(
        **defaults,
        context=context,
        messages=messages
    )
    palm_response = modify_palm_string(response.last)

    return palm_response

async def summarize_content(content: str, length: int, summarization_style: str = "Business Casual") -> str:
    context = f"Give an overview of the following news story or technological breakthrough. REMEMBER TO BE ENGAGING AND BE A STORYTELLER! Do not mention that you are a writer or that you are summarizing the article, just summarize the articles. Summarize the article in {length} words. Summarize the articles in the following tone: {summarization_style} "
    prompt = f"Summarize the following articles in a {summarization_style} tone with {length} words: {content}: "

    logging.info("Sending Request to llms")

    response = await replicate_summarization(content, length, context, prompt)
    if not response:
        response = await openai_summarization(content, length, context, prompt, summarization_style)
    if not response: 
        response = await palm_summarization(content, length, context, prompt, summarization_style)

    logging.info(f"Response Type: {type(response)}")

    return response



# def synthesize_speech_sync(speech_client, synthesis_input, voice, audio_config, index):
#     response = speech_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
#     temp_filename = f'/tmp/podcast_clip_{index}.wav'
#     with open(temp_filename, "wb") as temp_file:
#         temp_file.write(response.audio_content)
#     return temp_filename

# Asynchronous wrapper function
async def generate_podcast_clip(podcast_str, index):
    # Prepare the request payload
    data = {
        "input": {
            "text": podcast_str
        },
        "voice": {
            "languageCode": "en-US",
            "name": "en-US-Neural2-I",
            "ssmlGender": "MALE"
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16"
        }
    }

    # Set up the headers with the authorization token
    headers = {
        "Authorization": "Bearer " + google_access_token,
        "Content-Type": "application/json; charset=utf-8",
        "x-goog-user-project": google_project
    }

    # API endpoint
    tts_url = "https://texttospeech.googleapis.com/v1/text:synthesize"

    # Send the POST request asynchronously
    async with aiohttp.ClientSession() as session:
        async with session.post(tts_url, headers=headers, json=data) as response:
            print(response.status)
            print(response.text())
            if response.status == 200:
                temp_filename = f'/tmp/podcast_clip_{index}.wav'
                response_data = await response.json()
                audio_content = base64.b64decode(response_data['audioContent'])
                with open(temp_filename, "wb") as out:
                    out.write(audio_content)
                return temp_filename
            else:
                print("Error:", response.status, response.text())
                return None

async def process_podcasts(texts):
    
    tasks = [generate_podcast_clip(text, index) for index, text in enumerate(texts)]
    return await asyncio.gather(*tasks)


def generate_intro(audio_file1, audio_file2):
    """
    Fades out the first audio file after 20 seconds and overlays the second audio file at the same point.

    Parameters:
    audio_file1 (str): Path to the first audio file.
    audio_file2 (str): Path to the second audio file.

    Returns:
    AudioSegment: The combined audio segment.
    """
   # Load the first audio file
    audio1 = AudioSegment.from_file(audio_file1)

    # Parameters for the fade
    fade_start = 5 * 1000  # 5 seconds in milliseconds
    fade_end = 10 * 1000    # 10 seconds in milliseconds

    start_intro = 8 * 1000

    # Apply fade-out to the first audio file from 20 to 30 seconds
    audio1_faded = audio1[:fade_start] + audio1[fade_start:fade_end].fade_out(fade_end - fade_start)

    # Load the second audio file
    audio2 = AudioSegment.from_file(audio_file2)

    # Overlay the part of the second audio file onto the first, starting at the fade_start mark
    overlay_duration = fade_end - start_intro
    combined_audio = audio1_faded.overlay(audio2[:overlay_duration], position=start_intro)

    # Append the remaining part of the second audio file
    combined_audio = combined_audio + audio2[overlay_duration:]

    # Export the combined audio
    combined_audio.export("/tmp/intro.wav", format="wav")

    return combined_audio

async def process_all_articles(articles, target_word_count, tone):
    logging.info("Processing Articles Kickoff")
    tasks = [process_article(article, target_word_count, tone, index) for index, article in enumerate(articles)]
    
    processed_articles = await asyncio.gather(*tasks)
    print(tasks)
    return processed_articles

async def process_article(article_content, target_word_count_per_article, tone, index):
    link = article_content["link"]
    title = article_content["title"]
    content = article_content["content"]
    logging.info(f"Processing Article: {title}")

    summarized_content = await summarize_content(content, target_word_count_per_article, tone)
    podcast_clip_file = await generate_podcast_clip(summarized_content, index)

    return podcast_clip_file


def combine_clips(num_articles, transition_drum):
    combined_podcast = None
    for i in range(num_articles):
        clip_file = f'/tmp/podcast_clip_{i}.wav'
        clip = AudioSegment.from_wav(clip_file)
        if combined_podcast is None:
            combined_podcast = clip
        else:
            combined_podcast += clip
        combined_podcast += transition_drum
        os.remove(clip_file)
    return combined_podcast
    
async def generate_podcast(articles: list, target_word_count_per_article: int, user_id: int, s3_filename: str, tone: str = "Business Casual"):
    article_content_list = get_content(articles)
    logging.info(f"Amount of articles to process for podcast: {len(article_content_list)}")

    # Collect texts for TTS
    summarization_tasks = [summarize_content(article["content"], target_word_count_per_article, tone) for article in article_content_list]
    # Await all tasks in parallel
    texts_for_tts = await asyncio.gather(*summarization_tasks)
    logging.info(f"Amount of podcast scripts from llms: {len(texts_for_tts)}")

    # Process podcasts in parallel
    podcast_clip_files = await process_podcasts(texts_for_tts)
    logging.info(f"Amount of podcast clips created from articles: {len(podcast_clip_files)}")

    # Generate intro and combine clips
    # Note: Previously, the index was missing in this call. Adding index=0
    intro_fn = await generate_podcast_clip(f"Hello! Its {datetime.now().strftime('%A, %B %d')}. You're listening to The Toast. Today we're covering: {', '.join([article['title'] for article in article_content_list])}.", -1)
    # combined_podcast = AudioSegment.from_wav("podcast-intro.wav")
    logging.info(f"Intro Filename: {intro_fn}")
    combined_podcast = generate_intro("podcast-intro.wav", intro_fn)
    transition_drum = AudioSegment.from_wav("drumbit.wav")

    # Sequentially combine all podcast clips
    for i, clip_file in enumerate(podcast_clip_files):
        clip = AudioSegment.from_wav(clip_file)
        combined_podcast += clip + transition_drum
        os.remove(clip_file)  # Clean up the individual clip file

    # Save the final combined podcast
    combined_podcast.export('/tmp/podcast.wav', format='wav')
    
    # Upload to S3
    logging.info("Uploading File to S3")
    s3_client.upload_file(Bucket="user-podcasts", Filename='/tmp/podcast.wav', Key=s3_filename)
    logging.info("Finished Uploading File to S3")
    logging.info(f"S3 Filename: {s3_filename}")

    return s3_filename

################# MAIN Functions ################# 
def handler(event=None, context=None):
    print("Event:", event)
    if event == {}:
        event = {}

        event["articleIdRecommendations"] = [{"topic":"Testing", "articleID":101462124}, {"topic":"Testing", "articleID":101461582}]
        event["targetWordCount"] = 200
        event["format"] = "Podcast"
        event["userId"] = 12345
        event["tone"] = "Joking"
        event["s3Path"] = f'{event["userId"]}/{int(time())}-podcast.wav'

    try:
        articles = event["articleIdRecommendations"] # Dictionary
        target_word_count = event["targetWordCount"]
        content_format = event["format"]
        user_id = event["userId"]
        tone = event["tone"]
        s3_filename = event["s3Path"]
    except:
        return {
            'statusCode': 403,
            'errorMessage': "Not Valid event type.",
            's3Path':"None"
        }

    
    if content_format == "Podcast":
        asyncio.run(generate_podcast(articles, target_word_count, user_id, s3_filename, tone))
        logging.info(f"Podcast Generation Time: {time() - start_time}")
        return {
                'statusCode': 200
            }

    elif content_format == "Healthcheck":
        logging.info(f"Podcast Generation Time: {time() - start_time}")
        return {
            'statusCode': 200
        }
        
    else: 
        logging.info(f"Podcast Generation Time: {time() - start_time}")
        return {
            'statusCode': 403,
            'errorMessage': "Not Valid Content Format.",
            's3Path':"None"
        }
    


if __name__ == "__main__":

    event = {}

    event["articleIdRecommendations"] = [{"topic":"Testing", "articleID":101462124}, {"topic":"Testing", "articleID":101461582}]
    event["targetWordCount"] = 200
    event["format"] = "Podcast"
    event["userId"] = 12345
    event["tone"] = "Joking"
    event["s3Path"] = f'{event["userId"]}/{int(time())}-podcast.wav'
    

    articles = event["articleIdRecommendations"] # Dictionary
    target_word_count = event["targetWordCount"]
    content_format = event["format"]
    user_id = event["userId"]
    tone = event["tone"]
    s3_filename = event["s3Path"]

    
    if content_format == "Podcast":
        asyncio.run(generate_podcast(articles, target_word_count, user_id, s3_filename, tone))
    
    logging.info(f"Podcast Generation Time: {time() - start_time}")

        