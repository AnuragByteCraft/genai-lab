import openai
import ollama

OPENAI_API_KEY = "ollama-key"
# OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_MODEL = "gemma3:12b"

# Choose whether to use OpenAI API or Ollama
USE_OPENAI = False  # Set to True for OpenAI, False for Ollama
OPENAI_MODEL = "gpt-oss:20b"

# Function to get AI response
def get_response(user_prompt, system_prompt="You are a helpful assistant."):
    response = ollama.chat(
        model=OLLAMA_MODEL,  # Change to any available model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response["message"]["content"]


# Function to get AI response
def get_response_for_messages(messages):
    response = ollama.chat(
        model=OLLAMA_MODEL,  # Change to any available model
        messages=messages,
    )
    return response["message"]["content"]


if __name__ == '__main__':
    import os
    test_dir = r"/Users/ananth/PycharmProjects/book_generator/ppt_images/chapter4"
    # name = "page_1.png"
    name = "image_002_slide_3.png"
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            'role': 'user',
            'content': """
            Describe the given image.
            """,
            'images': [os.path.join(test_dir, name)]
        }
    ]

    results = get_response_for_messages(messages)

    print(results)

