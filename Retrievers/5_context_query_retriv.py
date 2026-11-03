# Contextual Compressor Retriever
# This is a retriever that uses the context of the query to retrieve relevant documents from a vector store. It is designed to handle queries that require understanding the context in which they are asked, rather than just matching keywords.    

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from dotenv import load_dotenv

# Loading the dotenv 
load_dotenv()

# llm 
llm = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# embediing model 
embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')

# Vector store 

vector_store = FAISS.from_documents(
     embedding = embedding,
     documents=docs
)

# vector store as retrieval 

retriever = vector_store.as_retriever(kwargs= {'k':5})


# Query 

query = "What is photosynthesis? "


# Simple retriver result 
simple_retrieve = retriever.invoke(query)

# Compressor 

compressor = LLMChainExtractor.from_llm(llm)

# Now contextual retriever 
context_retrieve = ContextualCompressionRetriever(
     base_retriever=retriever,
     base_compressor=compressor
)


context_result = context_retrieve.invoke(query)


# The final results 

# Simple retrieve result 
for i,doc in enumerate(simple_retrieve):
     print(f" \n ---- Result {i+1} ---")
     print(doc.page_content)
     

#  context result 

print("\n * "*10)

for i,doc in enumerate(context_result):
     print(f" \n ---- Result {i+1} ---")
     print(doc.page_content)




