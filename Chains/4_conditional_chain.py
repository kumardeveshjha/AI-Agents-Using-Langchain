# Conditional chain consits the decsion making for an AI AGENT 

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

parser = StrOutputParser()


# Creating the pydantic class for the consitant sentiment sentiment 

class Feedback(BaseModel):
     
     sentiment: Literal['positive','negative'] = Field(description='Give the sentiment of the user feed back')


parser_2 = PydanticOutputParser(pydantic_object=Feedback)
# prompt 1 

prompt_1 = PromptTemplate(
     template='Give me the sentiment of the given feedback \n {feedback} \n {format_instruction}',
     input_variables=['feedback'],
     partial_variables={'format_instruction':parser_2.get_format_instructions()}
)

# prompt 2 Prompt ofr the postive response 

prompt_2 = PromptTemplate(
     template="Give the direct appropriate response for this positive feedback \n {feedback}",
     input_variables=['feedback']
)


# Prompt for negative response 
prompt_3 = PromptTemplate(
     template="Give the direct appropriate response for this negative feedback \n {feedback}",
     input_variables=['feedback']
)


sentiment_chain = prompt_1 | model | parser_2

# making the branch of the conditiona for creating the effective chians 

response_chain = RunnableBranch(
     (lambda x: x["sentiment"].sentiment == 'positive', prompt_2 | model |parser),
     (lambda x: x["sentiment"].sentiment == 'negative', prompt_3 | model | parser),
     RunnableLambda(lambda x: "Can't speciffy the sentiment of the feedback" )
)

# The final chain 

classifier_chain = {
    "feedback": lambda x: x["feedback"],
    "sentiment": sentiment_chain
}

chain = classifier_chain | response_chain


user_feedback = "The phone is terrible, it has low batterry life and have many blotwares"

result = chain.invoke({'feedback':user_feedback})

# print(result)

chain.get_graph().print_ascii()