from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage 

load_dotenv()

# 
model = ChatGoogleGenerativeAI(
     model='gemini-3.5-flash-lite',
     )

# chat history 
chat_history = [
     SystemMessage(content='Yo are a helpful Assistant')
]


# The loop that works as a chatbot 
while True:
     user_input = input("You: ")
     chat_history.append(HumanMessage(content=user_input))
     if user_input.lower() == "exit":
        break
     result = model.invoke(chat_history)
     print("AI: ",result.content[0]['text'])
     chat_history.append(AIMessage(content=result.content[0]['text']))
print(chat_history)

# while True:
#     user_input = input("You: ")

#     if user_input.lower() == "exit":
#         break

#     result = model.invoke(user_input)

#     print("AI:", result.text)
     