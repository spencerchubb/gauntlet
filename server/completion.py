from typing import List

import boto3
bedrock_runtime = boto3.client(
  service_name='bedrock-runtime',
  region_name='us-east-2',
)

def bedrock_completion(system_prompt: str, messages: List[dict]):
  for message in messages:
    message["content"] = [{"text": message["content"]}]
  response = bedrock_runtime.converse(
    modelId="arn:aws:bedrock:us-east-2:474668398195:inference-profile/us.meta.llama3-2-3b-instruct-v1:0",
    messages=messages,
    system=[{"text": system_prompt}],
    inferenceConfig={
      "maxTokens": 512,
      "temperature": 1,
      "topP": 0.9,
    },
  )
  return response["output"]["message"]["content"][0]["text"]
