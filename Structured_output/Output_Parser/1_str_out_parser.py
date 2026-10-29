

# Output parser basic model of passing the output to the llm itself 
# First genearte the output and same output used as input again to the same llm 

from unittest import result

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

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

# creating the prompt 

prompt_1 = template_1.invoke({'topic':"Blackhole"})

result = model.invoke(prompt_1)

prompt_2 = template_2.invoke({'topic':result.content[0]['text']})

result_1 = model.invoke(prompt_2)
print(result_1.content[0]['text'])