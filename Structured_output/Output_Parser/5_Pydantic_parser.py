# this is pydantic output parser it not only design the schema but also provides the validation 

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

# Creating the schema class 

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# 
class Person(BaseModel):
     name: str = Field(description="Please give the name from the required country")
     age:int = Field(gt=18,lt=60 ,description="Give the age of the person from the particular region")
     city:str = Field(description="Give the city from the region described")
     
parser = PydanticOutputParser(pydantic_object=Person)


# Creating the prompt template 

template = PromptTemplate(
     template="Give the name, age and city of the person from the {region} \n {format_instructions} region",
     input_variables= ['region'],
     partial_variables=({"format_instructions":parser.get_format_instructions()})
     
)

# prompt = template.format(**{'region':"India"})

# result = model.invoke(prompt)

# output = parser.parse(result.content[0]['text'])

# print(output)

# using the chians of langchain 

chain = template | model | parser
result = chain.invoke({'region':'India'})

print(result)
