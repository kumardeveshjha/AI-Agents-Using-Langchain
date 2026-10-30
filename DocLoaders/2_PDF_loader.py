from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()



# Docloader 

# This will only work with the text data in the PDF, there are other PDF loaders 
loader = PyPDFLoader('GoogleMap.pdf')


# loading the doc 
doc = loader.load()

# storing the doc

print(len(doc))
# forming the chain 
# chain = prompt | model | parser



