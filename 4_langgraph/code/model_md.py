"""
model for generating and reviewing code using RAG
"""
import json
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from ingest_md import EMBEDDINGS_MODEL, DB_CHROMA_PATH

# ----------------------------------- Define LLM endpoint URLs -----------------------------------
# define the end point URL for LLM server - e.g. LMSTUDIO_URL = "http://localhost:1234/v1"
# also define a model name for the model you deployed on LMSTUDIO_URL
from ip_config import LMSTUDIO_PC_URL, model_name
# -----------------------------------------------------------------------------------------------

device = "cuda"  # use "mps" for mac

custom_prompt_template = """
You are an assistant for code generation tasks using the documentation on LangGraph.

You are given a set of retrieved context chunks. Each chunk includes:
- A part of documentation on a LangGraph topic, this may contain python code or text content
- Metadata fields: 
  • `source` (full file path)

Your objectives are:

1. If the question pertains to understanding concepts behind a LangGraph topic, provide detailed answer with examples
2. If you are asked for source code that may be intended to be executed automatically, provide only the code enclosed
   within ```python and ```. Do not provide any other explanation.

The following LLM is available to you:
1. A locally hosted LLM by LMStudio that is accessible at: http://localhost:1234/v1. 
When using an LLM you must use this as base_url parameter. The API_KEY must be set to "lm-studio".
2. The model name to use is: gemma-3-12b-it
3. A sample code to generate an LLM client is:

# Configure the local LLM with LM Studio
llm = ChatOpenAI(
    model="gemma-3-12b-it",
    base_url="http://localhost:1234/v1",  # LM Studio endpoint
    api_key="lm-studio",  # API key as specified
    temperature=0.001,
)

4. The development environment should be LangGraph based, do not use any other library like autogen or crewai for implementing agents.
5. Do not generate code that consists any deprecated functions, like create_react_agent. You must use the provided context and not hallucinate.
6. Do not assume availability of any external tool. Your code can assume python 3.11 library.
7. When you generate code, also provide an example invocation under the main program. Use __main__ to determine the main.
8. Do not use checkpointer unless explicitly requested.
9. Define the state as a TypedDict


Question: {question}
Context: {context}
Answer:
"""

# Example variables:
# question -> user question text
# context  -> concatenated retrieved context chunks with metadata


def set_custom_prompt():
    prompt = PromptTemplate.from_template(template=custom_prompt_template)
    return prompt


# def format_docs(docs):
#     for doc in docs:
#         print("#" * 100)
#         print(doc)
#     return "\n\n".join([d.page_content for d in docs])


def format_docs(docs):
    context = ""
    for doc in docs:
        pc = doc.page_content
        meta = doc.metadata

        print("=> ", pc)
        print("+++: ", meta)
        print("-" * 100)
        # combined = "\n\nChunk Content: " + pc + "\nMetadata: " + json.dumps({"source": meta["source"]}) + "\n"
        # context = context + combined

        context += pc

        # print("#" * 100)
        # print(doc)
    # print(" => ", context)
    # print("-" * 100)
    return context


def load_llm(base_url=None, api_key=None, temperature=0.00001, max_tokens=10000):
    """
    Provide an instance of a LLM created using LangChain library that adheres to OpenAI protocol.
    :param base_url: This is the endpoint where the LLM is being hosted
    :param api_key: API key used to authenticate with the LLM - for LM Studio this is any arbitrary non-empty string
    :param temperature: Lower numbers between 0.0 to 0.5 provide more deterministic results, > 0.5 creative outputs.
    :param max_tokens: Maximum number of tokens to use when generating the output.
    :return: a client instance for the LLM
    """
    if base_url is None:  # default is a localhost URL that serves the LLM
        base_url = "http://localhost:1234/v1"
        api_key = "lm_studio"

    if api_key is None: api_key = "lm_studio"

    return OpenAI(base_url=base_url, api_key=api_key, temperature=temperature, max_tokens=max_tokens, model="google/gemma-3-12b")


def load_llm_remote(base_url=LMSTUDIO_PC_URL, api_key=None, temperature=0.00001, max_tokens=10000):
    """
    A helper function to load a remote LLM instance
    :return: a client instance for the LLM running remotely
    """
    return OpenAI(base_url=base_url, api_key=api_key, temperature=temperature, max_tokens=max_tokens, model=model_name)


def get_retriever():
    """
    after texts are ingested in vectordb, get it as a retriever
    :return:
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,
                                       model_kwargs={'device': device})
    vectordb = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)
    return vectordb


# ---------------------------------------- RAG Entry Point for tool call -----------------------------------
def do_rag(query, base_url=None, api_key=None, temperature=0.00001, max_tokens=10000):
    """
    Given a query, perform Naive RAG using the vector database and return the result
    :param query: query to be executed
    :param base_url: This is the endpoint where the LLM is being hosted
    :param api_key: api key to authenticate with the LLM
    :param temperature: temperature setting - 0.0 means least variety, > 0.5 means higher variety
    :param max_tokens: Maximum number of tokens to use when generating the output.
    :return: results from Naive RAG
    """
    # 1. get a reference to the vector database using get_retriever function and cast it as retriever
    vectordb = get_retriever()
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    # 2. Get an instance of LLM using load_llm() or load_llm_remote
    # llm = load_llm(base_url=base_url, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    llm = load_llm_remote()

    print("LLM Loaded: ", llm)
    retrieved_docs = retriever.invoke(query)

    # Uncomment the lines below for debugging
    # print("Num retrieved docs = ", len(retrieved_docs))
    # print(retrieved_docs[0].page_content)

    # 3. Format the list of retrieved documents as a cohesive context, you can include metadata suitably
    context = format_docs(retrieved_docs)

    # 4. Given the context form a complete prompt by including other instructions
    prompt = set_custom_prompt()
    prompt = prompt.format(context=context, question=query)

    # 5. Execute the prompt on the LLM
    result = llm.invoke(prompt)

    return result


# ------------------------- RAG Test with command line --------------------------
def qa_bot():
    query = ""
    while query != "quit":
        query = input("Your Query: ")
        if query == "quit":
            break
        result = do_rag(query)
        print(result)


if __name__ == '__main__':
    qa_bot()

