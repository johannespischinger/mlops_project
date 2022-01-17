#! /bin/sh

gcloud ai-platform jobs submit training test_1 \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --model-dir=gs://$BUCKET_NAME/$MODEL_DIR
