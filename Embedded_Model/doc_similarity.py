
# This is very useful in making the embeddings of the text doc data and use it to make an ai app/ ai agent mostly RAG application

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2",output_dimensionality=50)

doc = [
     "The Venture capital are the type of firms that helps the startups ",
     "The startups that has only idea are known as early-stage ideas",
     "The profitability is highly evaluated during the funding time and fund return time",
     "Most of the VC funds burn money, because they don't have a clear picture about the vision and future traction",
     "If a VC wanted to increase it's success percentage it must focus on the future traction.",
     "he is farmer and ploughing his crop"
]


query= "I wanted to know how a VC evaluates the profitability"

embed_doc = embeddings.embed_documents(doc)
query_embd = embeddings.embed_query(query)

# checking cosine similarity score 


# Now checking the similarity score for the embeddings 

score= cosine_similarity([query_embd],embed_doc)[0]
similarity = sorted(enumerate(score),key=lambda x:x[1])
index, score = sorted(list(enumerate(score)),key= lambda x:x[1])[-1]
print(similarity)
print(query)
print(doc[index-1])
print("Similarity Score is :", score)

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity

# load_dotenv()

# embed_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2',output_dimensionality=10)

# # creating a doc 
# doc = [
#      "A data scientist is an engineer who has knowledge of data and machine learning.",
#      "To become a data scientist, you need to be good at data analysis and applying the machine learning algorithms",
#      "A data scientist uses these ML algorithms to take insights from data and make better decisions",
#      "These decisions from data are commonaly known as the predictions of ML model ",
#      "A good data scientist has also a good mathematical foundation specially in statistics and linear algebra"
# ]

# # User query 
# query = "How to become a data scientist"

# # converting the query and doc into the embedding 
# embed_doc = embed_model.embed_documents(doc)
# embed_query = embed_model.embed_query(query)

# # Now checking the similarity of the doc

# score = cosine_similarity([embed_query],embed_doc)[0]
# similarity = sorted(enumerate(score),key= lambda x:x[1])

# index,score = sorted(list((enumerate(score))),key= lambda x:x[1])[-1]

# print()
# print(query)
# print(doc[index])
# print("Similarity Score: ", score)
