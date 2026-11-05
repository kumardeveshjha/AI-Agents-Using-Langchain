from chromadb import Documents
from faiss import Embedding
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.rate_limiters import InMemoryRateLimiter
from regex import template
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()

# Model 
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# embeding 
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1.3, 
    check_every_n_seconds=0.1, 
    max_bucket_size=10
)
embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')

# The Youtube transcript  video id 
video_id = "wTsNjvuTi4U"

# Youtube transcript api 
transcript = YouTubeTranscriptApi()

try:
     # Mkaing the transcripts from the youtube 
     trans_list = transcript.list(video_id).find_transcript(['en']).fetch()
     # creating the chunks as a transcripted text
     transcripts = " ".join(chunk.text for chunk in trans_list)
     # print(transcripts)
except TranscriptsDisabled:
     print('Could not find the transcript')


# Now we are doing the text spliting 

splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

chunks = splitter.create_documents([transcripts])

# print(chunks)

# creating the embeddings of the given chunks and store them in the vector store 

vector_store = FAISS.from_documents(
     embedding=embedding,
     documents = chunks,

)

# checking the ids of the docs
# ids = vector_store.index_to_docstore_id

# retriever "Retrieval State of RAG"

retriever = vector_store.as_retriever(serch_type='similarity', kwargs={'k':3})

# Query 

query = "Name of the banks JP morgan acquired ?"

# context from the vector store 

context = retriever.invoke(query)

# creaiting a prompt template 

prompt = PromptTemplate(
     template=""" You are an expert financial analyst. 
                  Answer the following question from the follwing context only.
                  If you think that the given context is not sufficient to answer the question, simply say i do not have enough context.
                  {Context}
                  question: {query}
                """,
     input_variables=['Context','query']
     
)

final_context = '\n\n'.join(doc.page_content for doc in context)

# this is Augmentation step of RAG 

final_prompt =  prompt.format(**{"Context":final_context,"query":query})



# Generation of the final output 

result = model.invoke(final_prompt)

print(result)






