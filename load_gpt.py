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

def run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000, use_16k=False, wait_time = 1, temperature=1.0):

    completion = None
    
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

    while completion is None:
        wait_time = wait_time * 2
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
                temperature = temperature,
                max_tokens = num_tokens_request,
                n=num_gen,
                messages = messages
            )
            
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
    
    return completion.choices[0].message.content
