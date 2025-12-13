"""
OpenAI protocol interface - works with LM Studio, Ollama, DeepSeek, OpenAI, grok, etc
"""
import time
from openai import OpenAI
#
# from core.config import LM_STUDIO_URL, LM_STUDIO_API_KEY
#
# from core.config import DEEPSEEK_URL, DEEPSEEK_FIRST_API_KEY
# from core.config import DEEPSEEK_CHAT, DEEPSEEK_REASONER  # deepseek models
#
# from core.config import OPEN_ROUTER_BASE_URL, OPEN_ROUTER_DEEPSEEK_R1_API_KEY, OPEN_ROUTER_DEEPSEEK_R1_MODEL_NAME
# from core.config import GEMINI_FLASH_URL, GEMINI_FLASH_API_KEY, GEMINI_FLASH_MODEL
#
# from core.config import OPENAI_API_URL, OPENAI_API_KEY, OPENAI_MODEL
#

# --------------------- Define base_url and api_key ----------------------
# base_url = GEMINI_FLASH_URL  # DEEPSEEK_URL  # LM_STUDIO_URL
# api_key = GEMINI_FLASH_API_KEY  # DEEPSEEK_FIRST_API_KEY # LM_STUDIO_API_KEY
# model = GEMINI_FLASH_MODEL  # DEEPSEEK_CHAT  # r"gemma-2-9b-it"

# base_url = OPENAI_API_URL
# api_key = OPENAI_API_KEY
# model = OPENAI_MODEL  # r"gemma-2-9b-it"


# base_url = "http://192.168.68.107:1234/v1"  # LM_STUDIO_URL

base_url = "http://localhost:1234/v1"
api_key = "LM_STUDIO_API_KEY"
# model = r"gemma-3-12b-it"
model = "google/gemma-3-12b"
# model = "openai/gpt-oss-20b"
# model = "openai/gpt-oss-120b"
# base_url = OPEN_ROUTER_BASE_URL
# api_key = OPEN_ROUTER_DEEPSEEK_R1_API_KEY
# model = OPEN_ROUTER_DEEPSEEK_R1_MODEL_NAME


# ------------------------------------------------------------------------
# create a client
client1 = OpenAI(base_url=base_url, api_key=api_key)  # html to json


# def get_completion_messages(messages, client=client1, model=model, max_tokens=8000):
#     """
#     given the prompt, obtain the response from LLM hosted by LM Studio as a server
#     :param messages: messages for the LLM that contain the prompts
#     :return: response from the LLM
#     """
#     completion = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0.0,
#         max_tokens=max_tokens,
#         # stream=False,
#         # stream_options={"include_usage": True},
#     )
#     return completion.choices[0].message.content


def get_completion_messages(messages, client=client1, model=model, max_tokens=32000, temperature=0.0):
    """
    Get the completion text from the LM Studio LLM server.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ LLM call failed: {e}")
        return ""


def get_chat_completion_stream(messages):
    """
    Streams back tokens for the given messages.
    Yields chunks of text.
    """
    stream = client1.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if __name__ == '__main__':
    t1 = time.time()
    prompt = """
    Write a 3 page essay on Reinforcement Learning
    """
    messages = [
        {
            "role": "system",
            "content": "You are a professor in computer science."
        },

        {
            "role": "user",
            "content": prompt,
        }
    ]

    results = get_completion_messages(messages)
    print(results)
    t2 = time.time()
    print("Time taken: ", t2 - t1)

    # input("Enter any key to continue: ")
    #
    # response_text = ""
    # for token in get_chat_completion_stream(messages):
    #     response_text += token
    #     # print(token)
    #     print(response_text)
