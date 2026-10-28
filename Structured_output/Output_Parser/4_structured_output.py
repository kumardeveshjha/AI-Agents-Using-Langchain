from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import (
    ResponseSchema,
    StructuredOutputParser,
)


# This is used to give the schema to the json and also ity helps in providing the output_parser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# REsponse schema 

schema = [
     ResponseSchema(name="fact1",description="Tell me about the first fact about the given topic"),
     ResponseSchema(name="fact2",description="Tell me about the second fact about the given topic"),
     ResponseSchema(name="fact3",description="Tell me about the third fact about the given topic")
]

# JSON Structured  Parser 

parser  = StructuredOutputParser.from_response_schemas(schema)

# Template

template = PromptTemplate(
     template='Give the  3  most interesting facts about {topic} \n {format_instructions} ',
     input_variables=['topic'],
     partial_variables=({'format_instructions':parser.get_format_instructions()})

)

# creating a prompt from the template 

# prompt = template.format({'topic':'Blackhole'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content[0]['text'])

# print(final_result)

chain = template | model | parser

result = chain.invoke({"topic":"blackhole"})

print(result['fact1'])