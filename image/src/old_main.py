from google.cloud import texttospeech
import google.generativeai as palm
import openai
import replicate
import boto3
from pydub import AudioSegment
import requests

import shutil
import re
import asyncio
from time import time, sleep 
import os
import logging
import json
from datetime import datetime

################# Config #################


# Open AI API
print("Outputting API Keys: ") 
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")


# Replicate API 
replicate_token = os.getenv("REPLICATE_API_TOKEN")


# Palm API
palm_api_key = os.environ.get("PALM_API_KEY")
palm.configure(api_key=palm_api_key)
defaults = {
  'model': 'models/chat-bison-001',
  'temperature': 0.25,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
}

# TTS Model
speech_client = texttospeech.TextToSpeechClient()

# S3
logging.info("Connecting to s3")
s3_client = boto3.client('s3')
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


def replicate_summarization(content: str, length: int, context: str, prompt: str,  max_retries: int = 3):
    backoff_factor = 1.05
    token_buffer = 100
    model = "mistralai/mixtral-8x7b-instruct-v0.1"

    max_tokens = count_words(context) + count_words(content) + count_words(prompt) + length + token_buffer

    for attempt in range(max_retries):
        try:
            start_time = time()
            response = ''

            for event in replicate.stream(
                model,
                input={
                    "prompt":context + prompt + content,
                    "top_k": 50,
                    "top_p": 0.9,
                    "temperature": 0.6,
                    "max_new_tokens": length+token_buffer,
                    "presence_penalty": 0,
                    "frequency_penalty": 0
                },
            ):
                response += str(event)
            
            latency = time() - start_time
            failure = False
            return response
        except Exception as e:
            logging.warning(f"Replicate error occurred: {e}, attempt {attempt + 1} of {max_retries}")
            sleep(int(backoff_factor ** attempt))

        return False

def openai_summarization(content: str, length: int, context: str, prompt: str, summarization_style: str = "Business Casual", max_retries: int = 3):
    backoff_factor = 1.5
    token_buffer = 100

    max_tokens = count_words(context) + count_words(content) + count_words(prompt) + length + token_buffer
    print("Max Tokens:", max_tokens)
    if max_tokens > 16000:
        max_tokens = 16000

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"  # Replace with your actual API key
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
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logging.warning(f"OpenAIError occurred: {e}, attempt {attempt + 1} of {max_retries}")
            sleep(int(backoff_factor ** attempt))

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

def summarize_content(content:str, length:int, summarization_style:str="Business Casual") -> str:
    context = f"Give an overview of the following news story or technological breaktrhough. REMEMBER TO BE ENGAGING AND BE A STORYTELLER! \
                Do not mention that your are writer or that you are summarizing the article just summarize the articles. Summarize the article in {length} words. \
                     Summarize the articles in the following tone: {summarization_style} "
    prompt = f"Summarize the following articles in a {summarization_style} tone with {length} words: {content}: "
    
    
    response = replicate_summarization(content, length, context, prompt)
    if not response:
        response = openai_summarization(content, length, context, prompt, summarization_style)
    elif not response: 
        response = palm_summarization(content, length, context, prompt, summarization_style)

    logging.info(f"Response Type: {type(response)}")
    
    return response


def generate_podcast_clip(podcast_str:str, index:int) -> str:
    selected_voice_name = "en-US-Neural2-I"

    # send request to the Google TTS API
    synthesis_input = texttospeech.SynthesisInput(text=podcast_str)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=selected_voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = speech_client.synthesize_speech(
        input=synthesis_input, 
        voice=voice, 
        audio_config=audio_config
    )

    temp_filename = f'/tmp/podcast_clip_{index}.wav'
    with open(temp_filename, "wb") as temp_file:
        temp_file.write(response.audio_content)

    print(f"Finished Processing Article to file index : {index}")
    return temp_filename

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

    summarized_content = summarize_content(content, target_word_count_per_article, tone)
    podcast_clip_file = generate_podcast_clip(summarized_content, index)

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
    
def generate_podcast(articles:list, target_word_count_per_article:int, user_id:int, s3_filename:str, tone:str="Business Casual"):
    article_content_list = get_content(articles)
    combined_podcast = None
    index = 0

    num_articles = len(article_content_list)
    titles = ''
    for i in range(num_articles):
        article = article_content_list[i]

        title = article["title"]

        if num_articles > 1 and i == (num_articles-1):
            titles += " and "
        titles = titles + title + ". "
    
    now = datetime.now()

    # Mapping integers to weekdays
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Mapping integers to month names
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


    generate_podcast_clip(f"Hello! Its {weekdays[now.weekday()]}. {months[now.month - 1]} {now.day}. You're listening to The Toast. Today we're covering: {titles} Let's ride!", 0)
    generate_intro("podcast-intro.wav", "/tmp/podcast_clip_0.wav") # This generates the intro .wav file.

    combined_podcast = AudioSegment.from_wav("/tmp/intro.wav") 
    transition_drum = AudioSegment.from_wav("drumbit.wav")
    

    clip_files = []
    processed_articles = asyncio.run(process_all_articles(article_content_list, target_word_count_per_article, tone))

    # Sequential combination of clips
    combined_podcast += combine_clips(num_articles, transition_drum)

    filename = '/tmp/podcast.wav'
    s3_bucket = "user-podcasts"
    

    logging.info("Saving Audio to local")
    combined_podcast.export(filename, format="wav")
    logging.info(f"Combined Audio Saved to {filename}")

    logging.info(f"Saving podcast to s3 for user: {user_id} ")
    s3_client.upload_file(Bucket=s3_bucket, Filename=filename, Key=s3_filename)
    logging.info(f"Successfully saved podcast to s3 for user: {user_id}")

    return s3_filename

def handler(event=None, context=None):

    logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(processName)s] [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
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
        generate_podcast(articles, int(target_word_count), user_id, s3_filename, tone)
        return {
                'statusCode': 200
            }
    elif content_format == "Healthcheck":
        return {
            'statusCode': 200
        }
        
    else: 
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
        generate_podcast(articles, int(target_word_count), user_id, s3_filename, tone)
        