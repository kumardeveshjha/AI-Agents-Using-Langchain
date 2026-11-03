from langchain_core.documents import Document 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from dotenv import load_dotenv

load_dotenv()

# Model for embedding 
embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')


# docs 

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# vector store using the FAISS DB
vector_store = FAISS.from_documents(
     embedding=embedding,
     documents=all_docs
)  


# Sementic retrival using MMR

retrieval = vector_store.as_retriever(
     search_type='mmr',
     search_kwargs={'k': 3, 'lambda_mult': 0.5}
)

# query 
query = "What are the health benefits of walking and consuming leafy greens?"


# retrieval_result = retrieval.invoke(query)
# print(retrieval_result)


# sementic retrieval 




