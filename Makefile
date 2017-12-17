##This is a simple makefile helping to run the gcloud commands
##============================================================


help:          ## show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

local:         ## train locally
	gcloud ml-engine local train \
		--job-dir /code/output/${MODEL_NAME}_${JOB_SUFFIX} \
		--package-path trainer/ \
		--module-name trainer.task \
		-- \
		--epochs 3 \
		--input-dir /code/data/

serving:       ## serve locally
	tensorflow_model_server --port=9000 \
		--model_name=${MODEL_NAME} \
		--model_base_path=/code/output/${MODEL_NAME}_${JOB_SUFFIX}

upload-data:   ## upload train data
	gsutil cp -r /code/data/ gs://${BUCKET_NAME}/data/${MODEL_NAME}/

train:         ## train in the cloud
	gcloud ml-engine jobs submit training ${MODEL_NAME}_${JOB_SUFFIX} \
		--runtime-version ${RUNTIME_VERSION} \
		--job-dir gs://${BUCKET_NAME}/output/${MODEL_NAME}_${JOB_SUFFIX} \
		--package-path trainer/ \
		--module-name trainer.task \
		--region ${REGION} \
		-- \
		--epochs 3 \
		--input-dir gs://${BUCKET_NAME}/data/${MODEL_NAME}/

logs:          ## tail the cloud training log
	gcloud ml-engine jobs stream-logs ${MODEL_NAME}_${JOB_SUFFIX}

versions:
	gsutil ls gs://${BUCKET_NAME}/output/${MODEL_NAME}_${JOB_SUFFIX}/

version:       ## create a new version
	gcloud ml-engine versions create ${VERSION} \
        --runtime-version ${RUNTIME_VERSION} \
        --model ${MODEL_NAME} \
        --origin gs://${BUCKET_NAME}/output/${MODEL_NAME}_${JOB_SUFFIX}/${MODEL_BINARIES_JOB_TIMESTAMP}

inference:     ## predict in batch
	gcloud ml-engine jobs submit prediction ${MODEL_NAME}_${INFERENCE_SUFFIX} \
        --model ${MODEL_NAME} \
        --version ${VERSION} \
        --data-format TEXT \
        --region ${REGION} \
        --input-paths gs://${BUCKET_NAME}/data/${MODEL_NAME}/ \
        --output-path gs://${BUCKET_NAME}/output/${MODEL_NAME}_${INFERENCE_SUFFIX}

clean:         ## remove current job
	gsutil rm -r gs://${BUCKET_NAME}/output/${MODEL_NAME}_${JOB_SUFFIX}
