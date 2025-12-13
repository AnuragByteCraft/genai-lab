from langchain_openai import ChatOpenAI


def get_llm(temperature=0):
    """
    for the llm server running under LM Studio, get a Langchain llm client object.
    Use this if you are creating a langchain workflow that uses LLM as a local server.
    If you are not using langchain, use huggingface transformers to load the llm file or use get_completion() client.

    :param temperature: between 0 to 1, 0 for no creativity and 1 for maximum creativity due to variance
    :return:
    """
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        temperature=temperature,
        api_key="not-needed"
    )
    return llm


if __name__ == '__main__':
    llm = get_llm()
    response = llm.invoke("hi I am Ananth!")

    print(response)
    print(response.response_metadata)

