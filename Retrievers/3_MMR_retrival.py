# MMR stands for maximal margin relavance 
# This gives the relavant nad diverse results for the query 

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# creating docs 
docs = [
     Document(page_content="I am Deevesh Kumar from Agra Uttar Pradesh."),
     Document(page_content="My fathername is Narendra Kumar."),
     Document(page_content="I am a founder and investor of a fintech company."),
     Document(page_content="I am also a forbes 30 under 30 personality"),
     Document(page_content="Times magzine has published my interview two times"),
     Document(page_content="I have also a guest at top VC and fintech firms"),
     Document(page_content="My core phylosophy is the best personality has ability and character, where ability gives career and character make it longterm identity")
]


# embedding model 

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')


# Vector store using the chroma DB 

vector_store = FAISS.from_documents(
     embedding=embedding,
     documents = docs
     
)

# Converting to the retrieval 

retrieval = vector_store.as_retriever(
     search_type='mmr',
     search_kwargs={'k':3,'lambda_mult':0.5}
)


# query 
query = "What kind of firms i have been part"

# result 

result = retrieval.invoke(query)

print(result)