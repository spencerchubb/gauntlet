from typing import List, Literal
from pydantic import BaseModel

import boto3
bedrock_runtime = boto3.client(
  service_name='bedrock-runtime',
  region_name='us-east-1',
)

from groq import Groq
groq_client = Groq()

from openai import ChatCompletion, OpenAI
openai_client = OpenAI()

class Message(BaseModel):
  role: Literal["system", "user", "assistant"]
  content: str

def pretty_print_messages(messages: List[Message]):
  for message in messages:
    print(f"{message.role}: {message.content}")

def openai_compatible_completion(client, messages: List[Message], model, stream=False) -> ChatCompletion:
  return client.chat.completions.create(
    model=model,
    messages=messages,
    stream=stream,
    temperature=1,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
  )

def groq_completion(messages: List[Message], model: str = "llama3-8b-8192", stream=False) -> ChatCompletion:
  '''
  llama3-8b-8192: $0.05/$0.08  
  llama3-70b-8192: $0.59/$0.79  
  gemma-7b-it: $0.07/$0.07  
  mixtral-8x7b-32768: $0.24/$0.24  
  '''
  return openai_compatible_completion(groq_client, messages, model, stream)

def openai_completion(messages: List[Message], stream=False) -> ChatCompletion:
  return openai_compatible_completion(openai_client, messages, "gpt-4o-mini", stream)
