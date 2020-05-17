#!/usr/bin/env sh

echo "Deploying Production model name=SKLearnWeatherForestModel"

export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
mlflow models serve --model-uri models:/SKLearnWeatherForestModel/production --no-conda
