# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile
FROM python:3.9-slim


# Installs pytorch and torchvision.
RUN apt update && \
  apt install --no-install-recommends -y build-essential gcc && \
  apt clean && \
  apt-get install -y wget && \
  rm -rf /var/lib.apt/lists/*

WORKDIR /
# Copies the directories
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY /src /src
COPY /data/processed /data/processed
# RUN pip install torch torchvision

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
# RUN pip install cloudml-hypertune
COPY dtu-mlops-ac6fc7030842.json dtu-mlops-ac6fc7030842.json
ENV GOOGLE_APPLICATION_CREDENTIALS dtu-mlops-ac6fc7030842.json

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install torchvision


# Sets up the entry point to invoke the training file.
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

# export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
# export IMAGE_REPO_NAME=mnist_pytorch_custom_container
# export IMAGE_TAG=mnist_pytorch_cpu
# export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# docker build -f Dockerfile -t $IMAGE_URI ./