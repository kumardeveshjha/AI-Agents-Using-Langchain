from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# Prompt 

prompt = PromptTemplate(
     template = 'Giveme the best 5 funfact about {topic}',
     input_variables= ['topic']
)

# Parser 

parser = StrOutputParser()

# Making the chain 

chain = prompt | model | parser

# invoking the chain 

result = chain.invoke({'topic':'VC'})

# printing the result 

print(result)
