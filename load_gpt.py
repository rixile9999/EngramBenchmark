import getpass
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(("Enter your OpenAI API key: "))


openai_chat = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    timeout=5,
    max_retries=2,
)

openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
)

def get_openai_embedding(texts):
    return openai_embeddings.embed_documents(texts)

def run_openai_chat(prompt: str, messages):
    """
    Run an OpenAI chat completion.
    If `messages` is provided, it will be used; otherwise uses `prompt` as system message.
    """
    if messages is None:
        messages = [HumanMessage(content=prompt)]
    response = openai_chat.invoke(messages)
    return response.content


def run_chatgpt_with_examples(query, examples, input):
    messages = [
        {"role": "system", "content": query}
    ]
    for inp, out in examples:
        messages.append(
            {"role": "user", "content": inp}
        )
        messages.append(
            {"role": "system", "content": out}
        )
    messages.append(
        {"role": "user", "content": input}
    )   
    result = openai_chat.invoke(messages)
    return result.content
