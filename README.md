# Quickstart

## Install

    pip install -e .

## Set the variables

    export MILA_USERNAME=...  # auth for the login page
    export MILA_PASSWORD=...  # auth for the login page
    export OPENAI_API_KEY=...  # valid openai key to use
    export HUB_DATASET_ID=...  # link to the dataset hosted on huggingface
    export HUB_TOKEN=...  # huggingface api key

## Run the app locally

    gradio gradio_app.py --reload
