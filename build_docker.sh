#! /bin/sh

export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=mnist_pytorch_custom_container
export IMAGE_TAG=mnist_pytorch_cpu
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f gc_trainer.dockerfile -t $IMAGE_URI ./
docker push $IMAGE_URI