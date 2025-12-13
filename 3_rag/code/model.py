from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import EMBEDDINGS_MODEL, DB_CHROMA_PATH

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from core.ip_config import PC_BASE_URL, MAC_BASE_URL

# from get_llm import load_llm_client

custom_prompt_template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer precise but provide as much details as needed.
Question: {question} 
Context: {context} 
Answer: 
"""


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


def load_llm_remote(base_url=PC_BASE_URL, api_key=None, temperature=0.00001, max_tokens=10000):
    """
    A helper function to load a remote LLM instance
    :return: a client instance for the LLM running remotely
    """
    return OpenAI(base_url=base_url, api_key=api_key, temperature=temperature, max_tokens=max_tokens)


def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt


def format_docs(docs):
    """
    This is a function that is executed after retriever is invoked. The docs returned by the retriever are
    processed to extract the page_content strings and concatenated to construct the context variable.
    The output of this function is typically placed in a prompt, along with input query
    and sent to an LLM for final answer.
    :param docs: list of Langchain's document objects, each has metadata and page_content attributes.
    :return: string that is obtained by concatening page_content.
    """
    # you can process page_content as well as metadata
    # for doc in docs:  # for debugging and illustration we print the retrieved docs
    #     print(doc)
    print(docs)
    context_string = "\n\n".join([d.page_content for d in docs])
    # print(context_string)
    return context_string


# def load_llm_client():
#     """
#     Load the llm client that is compatible with Langchain.
#     Note that the client that is returned here is a langchain object and not a general openai client
#     :return: langchain client supporting openai protocol
#     """
#     client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")  # html to json
#     return client


def get_retriever():
    """
    after texts are ingested in vectordb, get it as a retriever
    :return:
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,
                                       model_kwargs={'device': 'cuda'})
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

