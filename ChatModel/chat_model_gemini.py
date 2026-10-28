# This is the main model which is mostly used and it is simple to make 

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
load_dotenv()

print("The AI Model Response")

start_time = time.time()

model = ChatGoogleGenerativeAI(model='gemma-4-26b-a4b-it')  #gemini-3.5-flash-lite  || gemini-3.8-flash /gemma-4-26b-a4b-it(fast 4B active parameters MoE) /gemma-4-31b-it(More better compute, Dense model all parameter active):
result = model.invoke("Can you build a schedule for me. My tasks are learning  machine learning, Doing Yoga, Learning about businesses. I have only 4 hours time")

# print(result.content_blocks[0]['text'])
print(result.content[0]['thinking']) # for gemma-4-26b-a4b-it model

end_time= time.time()

print("time taken: ",end_time - start_time)

