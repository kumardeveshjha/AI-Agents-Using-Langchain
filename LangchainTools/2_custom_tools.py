# This is about building our own custom tools 
# Why these are required 
# 1. When we wanted to interact with the APIs
# 2. We have our own database and wated to interact with it 
# 3. Encapsulate the business llogic

from typing import Required, Type
from langchain.tools import tool

# calucating the math 

@tool
def multiplier( a: int, b:int) -> int:
     """ Multiply a and b """
     return a*b


# print(multiplier.invoke({'a':2,'b':5}))
# print(multiplier.name)
# print(multiplier.description)



# 2 Tool Used for the custom tool building 
# This is called structuredTool class 
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Creating the datatype validation class 

class InputDataValid(BaseModel):
     a: int = Field(Required=True, description="Please use integer datatype arguemnts")
     b: int = Field(Required= True, description="Please use integer datatype arguemnts")


# My function to call 

# def multiply(a: int, b: int)->int:
#      return a*b

# # creating the tool to call the function 

# multi_tool = StructuredTool.from_function(
#      func=multiply,
#      name='multiply fuction',
#      description = "Tool to multiply two digits",
#      args_schema=InputDataValid
# )

# res = multi_tool.invoke(({'a':5,"b":10}))

# print(res)


# 3 The Third method is using the base class of the langchain tool 
from langchain_core.tools import BaseTool

class CustomTool(BaseTool):
     name: str = 'Multiply'
     description:str = "A tool to multiply two numbers"
     args_schema: Type[BaseModel] = InputDataValid 
     
     def _run(self, a:int,b:int)->int:
          return a*b


mul_tool = CustomTool()   
     
result = mul_tool.invoke({"a":5,"b":5})

print(result)



