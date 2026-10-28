from pydantic import BaseModel,EmailStr,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import  Annotated, Literal, Optional

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')


class Review(BaseModel):
     Brand: str = Field(description="Specify the model and brand of the phone from review")
     summary: str = Field(description="Give the summary of the review")
     sentiment: str = Field(description="Please find out the overall sentiment from the given review")
     pros: list[str]= Field(description="Find out the pros from the review as list")
     cons: list[str] = Field(description="Find out the cons from the review as list")
     specs: Optional[str] = Field(description="List all the specs about the phone")
     
     

structured_output = model.with_structured_output(Review) 

result = structured_output.invoke("""I have been using the Samsung Galaxy S26 for a while now, and overall, I am pretty satisfied with it. The phone looks premium and feels comfortable to hold. The 6.3-inch AMOLED display is bright and smooth, and the 120Hz refresh rate makes scrolling and using apps feel really nice. The performance is also great, with 12GB RAM and a powerful processor that handles multitasking, gaming, and everyday apps without much trouble. The 50MP main camera takes sharp photos in daylight, and the 3x optical zoom is useful, although low-light pictures could be better. Battery life is decent with normal use, but heavy usage can drain it before the end of the day, and charging speed is a bit disappointing for a phone at this price. I also like the software experience and useful AI features, but some of them do not feel very necessary in daily life. The biggest downside is the price, as the upgrades may not feel significant if you already own a recent Samsung flagship. Overall, it is a good premium phone with an excellent display, smooth performance, and reliable cameras, but I expected better battery life and faster charging. I would recommend it if you want a compact flagship phone, preferably at a discount.""")

print(result.pros)

