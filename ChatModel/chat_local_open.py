

# Use this whenever we are building a local open source model 
# Here main component is hfpipelines


from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
import os
os.environ['hf_home'] = "D:/hugging_face_cache"

llm = HuggingFacePipeline.from_model_id(
     model_id='Qwen/Qwen3-4B-Thinking-2507',
     task='text-generation',
     pipeline_kwargs= dict(
          temperature=1.8,
          max_new_tokens=100,
     )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the name of indian flag")

print(result.content)