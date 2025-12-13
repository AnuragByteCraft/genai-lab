import os

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from keys.my_keys import setup_environ

from agentic_ide_backend_enhanced import backend_main


setup_environ()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Agentic AI Course: Nov-Dec 2025"

BASE_URL = "http://192.168.68.122:1234/v1"
MODEL_NAME = "openai/gpt-oss-20b"

openai_client = wrap_openai(OpenAI(base_url=BASE_URL))


# This is the retriever we will use in RAG
# This is mocked out, but it could be anything we want
def retriever(query: str):
    results = [f"Your prompt is: {query}"]
    return results


# This is the end-to-end RAG chain.
# It does a retrieval step then calls OpenAI
def rag(question):
    return backend_main()
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:

    {docs}""".format(docs="\n".join(docs))

    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model=MODEL_NAME,
    )


if __name__ == '__main__':
    print(os.environ["OPENAI_API_KEY"])
    print(os.environ["LANGSMITH_API_KEY"])
    print(os.environ["LANGSMITH_TRACING"])
    # print(os.environ["LANGSMITH_WORKSPACE_ID"])

    response = rag("How to build a hello world agentic app with LangGraph?")
    # response = response.choices[0].message.content
    print(response)
