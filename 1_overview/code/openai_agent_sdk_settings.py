from agents import set_default_openai_key
from openai import AsyncOpenAI
from agents import set_default_openai_client

from agents import set_default_openai_api
from core.config import OPENAI_API_KEY, LM_STUDIO_REMOTE_URL, LM_STUDIO_API_KEY, LM_STUDIO_URL

import os

MY_URL = LM_STUDIO_URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Though we are using local LLM, we need OpenAI API key to avoid tracing related warnings, we are fine otherwise.
# OPENAI_API_URL = "https://api.openai.com/v1"
# OPENAI_MODEL = "gpt-4o"


def initialize_openai_agent_sdk():
    set_default_openai_key("lmstudio")
    custom_client = AsyncOpenAI(base_url=MY_URL, api_key=OPENAI_API_KEY)
    set_default_openai_api("chat_completions")
    set_default_openai_client(custom_client)
    return
