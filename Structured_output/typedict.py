# This is used when we are working with the LLMs , which has with_structured_output class

# This library helps in desgining the schema of the output dictionary, defining the datatypes of keys in dict.
# This only tells what datatype to put but not validates the datatype so there will be no error if we put wrong datatype. 

from typing import TypedDict

class Person(TypedDict):
     name:str
     age:int
     
new_person : Person = ({'name':"Devsh Kuamr","age":30})

print(new_person)