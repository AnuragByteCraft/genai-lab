"""
model for generating and reviewing code using RAG
"""
import json
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from core.session1_foundations.ingest_code import EMBEDDINGS_MODEL, DB_CHROMA_PATH

device = "mps"

# custom_prompt_template = """
# You are an assistant for question-answering tasks pertaining to Python source code.
# Use the following pieces of retrieved context to answer the question.
# If you don't know the answer, just say that you don't know.
# When a function definition is asked please provide the full function source code.
# Question: {question}
# Context: {context}
# Answer:
# """

custom_prompt_template = """
You are an assistant for question-answering tasks about a source code repository
that includes Python, Rust, and shell scripts.

You are given a set of retrieved context chunks. Each chunk includes:
- The code or text content
- Metadata fields: 
  • `source` (full file path)

Your objectives are:

1. Use both the code and any text to answer the user's question accurately and concisely.
2. Always include **traceability**:
   - Explicitly list which files your answer draws information from.
   - Use the file paths from the metadata exactly as provided.
3. When you cite information from a file, note it inline like:
   `(source: <file_path>)`.
4. At the end of the answer, include a **"Sources" section** summarizing all the files you referenced.
5. If the question asks for a function definition, reproduce the **entire function source code**.
6. If you do not know the answer or it cannot be found in the context, clearly state that.

Format your response exactly like this:

Answer:
<detailed answer using code snippets, inline citations, e.g. (source: /path/to/file.py)>

Sources: <Provide a maximum of upto 3 sources, do not repeat any source>
- <path to the source file> — <short summary or role>

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
        combined = "\n\nChunk Content: " + pc + "\nMetadata: " + json.dumps({"source": meta["source"]}) + "\n"
        context = context + combined
        # print("#" * 100)
        # print(doc)
    # print(context)
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


def load_llm_remote(base_url="http://192.168.68.108:1234/v1", api_key=None, temperature=0.00001, max_tokens=10000):
    """
    A helper function to load a remote LLM instance
    :return: a client instance for the LLM running remotely
    """
    return OpenAI(base_url=base_url, api_key=api_key, temperature=0.0, max_tokens=10000)


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
    llm = load_llm(base_url=base_url, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
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

