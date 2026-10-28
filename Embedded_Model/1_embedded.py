
# This is very useful because it gives the embedding of the test data which is further used for training and building an AI app

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import time
from dotenv import load_dotenv

load_dotenv()

start_time = time.time()

# Creating embeddings using gemini embedding models
embedding= GoogleGenerativeAIEmbeddings(model='gemini-embedding-2',output_dimensionality=20)

# Resulted embeddings 
result = embedding.embed_query("My name is Devesh Kumar")

# End time 
end_time = time.time()
print(result)
print(end_time-start_time)





