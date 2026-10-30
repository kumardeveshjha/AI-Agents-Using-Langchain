from langchain_community.document_loaders import WebBaseLoader
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
     template='Give me the answers of the questions: {questions} of the  {webpage_content}',
     input_variables=['questions','webpage_content']
)
# The website Url
url = "https://brianbalfour.com/essays/patreon-onboarding-growth"

# Loading the Website content 
loader = WebBaseLoader(url)

doc = loader.load()

# The webscrapped content 
content = doc[0].page_content

# Chains

chain = prompt | model | parser

# 
result = chain.invoke({'questions':"What was the best strategy of the patreaon",'webpage_content':content})

print(result)



