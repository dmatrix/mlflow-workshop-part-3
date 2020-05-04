#!/usr/bin/env sh

echo "Deploying model with Run ID=$1"

mlflow models serve --model-uri runs:/$1/model --no-conda
