from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Thinking-2507",
    task="text-generation",
    max_new_tokens=100,
   
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of india")


print(result.content)