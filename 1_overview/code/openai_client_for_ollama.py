from openai import OpenAI

base_url = "http://localhost:11434/v1"

client = OpenAI(
    base_url=base_url,  #
    api_key='ollama',  # required, but unused
)


def get_completion_ollama(messages, model="gpt-oss:20b"):
    """
    Given the prompt messages in OpenAI protocol, runs this as a client to Ollama and returns the results
    :param messages: list of messages constituting a prompt
    :param model: model to invoke
    :return: response (results) as a string
    """
    response = client.chat.completions.create(
      model=model,
      messages=messages
    )
    output = response.choices[0].message.content
    return output


if __name__ == '__main__':
    query = """
    Andrej Karpathy said recently: "The best programming language today is English".
    Do you agree?
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
        # {"role": "assistant", "content": "Yes there is a point in what he said."},
        # {"role": "user", "content": "Explain your answer in detail."}
    ]

    results = get_completion_ollama(messages)
    print(results)
