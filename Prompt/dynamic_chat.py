# This file consists of the type of providing the dynamic messages to the store chat history 
# This is also known as multi turn Messages

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# this is used to give the dynamic behavior to our system messages 

chat_template = ChatPromptTemplate(
     [('system','You are a helpful {role}.'),
     ('human','Help me to solve {task}')]
)

result = chat_template.invoke({
     "role":"researcher",
     "task":"question"
})

print(result)