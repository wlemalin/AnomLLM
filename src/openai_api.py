import os
from loguru import logger
from openai import OpenAI
import yaml


credentials = yaml.safe_load(open("credentials.yml"))


def openai_client(
    model,
    api_key=None,
    base_url="https://api.openai.com/v1"
):
    if api_key is None:
        assert model in credentials, f"Model {model} not found in credentials"
        api_key = credentials[model]["api_key"]
        if "base_url" in credentials[model]:
            base_url = credentials[model]["base_url"]
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    logger.debug(
        f"API key: ****{api_key[-4:]}, endpoint: {base_url}"
    )
    
    return client


def send_openai_request(
    openai_request,
    model,
    api_key=None,
    base_url="https://api.openai.com/v1"
):
    client = openai_client(model, api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model, **openai_request
    )
    return response.choices[0].message.content
