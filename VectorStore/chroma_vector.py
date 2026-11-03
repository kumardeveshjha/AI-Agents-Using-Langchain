

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv 

load_dotenv()


embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2",output_dimensionality=3072)

vector_store = Chroma(
     embedding_function=embedding,
     persist_directory='chroma_db',
     collection_name='cricket_data'
)


doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his bowling.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1,doc2,doc3,doc4,doc5]


# storing the documents in the vector store 

# vector_store.add_documents(docs)

# Printing the stored documents 

# print(vector_store.get())


# Updating the document on the vector store 

# doc = Document(
#      page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. He is known for his exceptional fielding skills.",
#      metadata={"team": "Chennai Super Kings"}
# )

# vector_store.update_document(document_id="99a7046a-a765-4c66-a50e-f7afd2a91f22",document=doc)


# Deleting the doc 

# vector_store.delete(ids="99a7046a-a765-4c66-a50e-f7afd2a91f22")
# print(vector_store.get())

# Adding a doc in the docy=uments 

# doc6 = Document(
#      page_content="Suryakumar Yadav is a stylish batsman known for his innovative stroke play and consistency in the IPL.",
#      metadata={"team": "Mumbai Indians"}
# )

# vector_store.add_documents(documents=[doc6])

# print(vector_store.get())

# Querying on the documents 

# best_batsmen = vector_store.similarity_search(
#      query="Give the best baller in IPL",
#      k=2 
# )

# print(best_batsmen)

# Query based on metadata 
print(vector_store.similarity_search(
     query="Give the batsman from mumbai indians",
     filter={"team":"Mumbai Indians"},
     k=3
))


# print(vector_store.get())