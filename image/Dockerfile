FROM public.ecr.aws/lambda/python:3.10

# RUN apt-get update && apt-get install -y curl gcc apt-transport-https lsb-release openssh-client git
RUN yum -y update
RUN yum install -y tar gzip gcc curl apt-transport-https gnupg ca-certificates

RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN gcloud --version

# gcloud installation:
# RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-456.0.0-linux-x86_64.tar.gz
# RUN tar zxvf google-cloud-cli-456.0.0-linux-x86_64.tar.gz
# RUN rm google-cloud-cli-456.0.0-linux-x86_64.tar.gz
# RUN ./google-cloud-sdk/install.sh
# RUN ./google-cloud-sdk/install.sh --screen-reader=true
# RUN ./google-cloud-sdk/bin/gcloud init
# RUN /google-cloud-sdk-/bin/gcloud --version

COPY charlie-tokenizedtoast-serviceAccount-GoogleAPI.json /image/service-account-key.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/image/service-account-key.json"
RUN gcloud auth activate-service-account --key-file /image/service-account-key.json

# PALM API load in api key file
ENV PALM_API_KEY=${PALM_API_KEY}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV REPLICATE_API_TOKEN=${REPLICATE_API_TOKEN}

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=us-east-1
ENV AWS_DEFAULT_OUTPUT=json

RUN pip install numpy pandas tqdm joblib boto3 awscli google-cloud-texttospeech google-generativeai openai pydub replicate

COPY src/* .
CMD ["main.handler"]
