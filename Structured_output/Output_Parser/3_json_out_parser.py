# this is to make a JSON style output parser 

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# JSON Parser 

parser = JsonOutputParser()

# Prompt Template

# this is most important part here because here we have given a static prompt and it can easily extract the schema from the prompt

template = PromptTemplate(
     input_variables=[],
     template = "Give me a indian finctional character's name age and city \n {formate_instructions}",
     partial_variables={'formate_instructions':parser.get_format_instructions()}
) 

# # creating the prompt from template
# prompt = template.format()

# # passing the prompt to the model
# result = model.invoke(prompt)

# # Now parsing the result for the better structured output in the form of JSON
# final_result = parser.parse(result.content[0]['text'])
# print(final_result)


# doing it using the chain method of langchain 

chain = template | model | parser 

result  = chain.invoke({})

print(result)