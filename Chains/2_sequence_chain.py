# Sequential chains keeps the more than one call from the model and form a sequence of the calls

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# prompt 1 

prompt_1 = PromptTemplate(
     template='Give me the detailed report about {topic}',
     input_variables=['topic']
)
prompt_2 = PromptTemplate(
     template='Give me the 5 pointer summary about {topic}',
     input_variables=['topic']
)

parser = StrOutputParser()

# Making a sequential chain 

chain = prompt_1 | model | parser | prompt_2 | model | parser

result = chain.invoke({'topic':"Cricket"})

# print(result)

chain.get_graph().print_ascii()
