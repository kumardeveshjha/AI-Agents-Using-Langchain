# This is used to make a well defined output parser 

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.8-flash')

# Prompt 1 

template_1 = PromptTemplate(
     input_variables=['topic'],
     template="Write a detailed report on {topic}"
)

# prompt_2 
template_2 = PromptTemplate(
     input_variables=['topic'],
     template="Give me a 5 line summary of this {topic} "
)

# Creating an output parser 

parser = StrOutputParser()

# creating a chian of output 

chain = template_1 | model| parser| template_2| model | parser 

# the final result of the structured output 

result = chain.invoke({'topic':"Blackhole"})

print(result)

