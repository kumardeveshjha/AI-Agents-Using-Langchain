
# This is the simplest way to loading the doc 

from langchain_community.document_loaders import TextLoader
# from langchain_core.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser




load_dotenv()

# Creating the model 
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# Parser 
parser = StrOutputParser()

# prompt 

prompt = PromptTemplate(
     template='Give the summary of the poem {poem}',
     input_variables=['poem']
)


# Docloader 
loader = TextLoader('my_VC.txt',encoding='utf_8')


# loading the doc 
doc = loader.load()

# storing the doc
poem = doc[0]

# forming the chain 
chain = prompt | model | parser

# Final_chain 

result = chain.invoke({'poem':poem})

print(result)


