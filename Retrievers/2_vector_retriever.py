from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# creating docs 
docs = [
     Document(page_content="I am Deevesh Kumar from Agra Uttar Pradesh."),
     Document(page_content="My fathername is Narendra Kumar."),
     Document(page_content="I am a founder and investor of a fintech company."),
     Document(page_content="I am also a forbes 30 under 30 personality"),
     Document(page_content="My core phylosophy is the best personality has ability and character, where ability gives career and character make it longterm identity")
]


# embedding model 

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')


# Vector store using the chroma DB 

vector_store = Chroma.from_documents(
     embedding=embedding,
     documents = docs,
     collection_name="my_data",
     
)


# Converting the vector store to the retriever 
retriever = vector_store.as_retriever(search_kwargs={"k":2})

# query 

query = "What is my name "

results = retriever.invoke(query)


for i,doc in enumerate(results):
     print(f"Result-{i+1}-------")
     print(results[i].page_content)


