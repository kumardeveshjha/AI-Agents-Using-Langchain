from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2",output_dimensionality=30)

doc = [
     "This is my document embedding using google gemini ",
     "I am using it in the free version",
     "I am also trying to make best of it ",
     "This will help me in building my first AI agent"
]

# here we are using embedded documents so i will 
embedded = embeddings.embed_documents(doc)

print(embedded)