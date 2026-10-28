from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st 
from langchain_core.prompts import PromptTemplate
load_dotenv()


# Creating the streamlit header 
st.header("Research Tools")

# Making the model using the google gemini model
model = ChatGoogleGenerativeAI(
     model='gemini-3.5-flash-lite',
     temperature=0.3,
     )

# Creating the the prompt template 
# This helps in making dynamic prompts 

template = PromptTemplate(
     input_variables=['paper_input','stype_input','length_input'],
     template= """
               Please summarize the research paper titled "{paper_input}".

               Explanation Style: {style_input}
               Explanation Length: {length_input}

               Cover:
               1. Research objective, problem, and contributions.
               2. Methodology and proposed approach.
               3. Important equations and technical concepts.
               4. Key experimental results and findings.
               5. Intuitive explanations and relatable analogies.
               6. Limitations and future work.
               7. Practical applications and takeaways.

               Accuracy rules:
               - Use only information supported by the paper.
               - Do not invent equations, results, or experimental details.
               - If information is missing, state "Insufficient information available."
               - Distinguish author claims from your interpretations.
               - Follow the requested style and length.
               """
)

# Imput vaiables from the user as drop downs 

# Paper type input
paper_input = st.selectbox(
     "Please select the paper:",
     ['Attention All you Need',
      'BERT: Pretraining of bidirectional Transformers',
      'Diffusion model Beats GAN']
)

# style input 

style_input = st.selectbox(
    "Explanation Style",
    [
        "Beginner-friendly and intuitive",
        "Simple language with analogies",
        "Technical and academic",
        "Step-by-step tutorial",
        "Mathematical and equation-focused",
        "Researcher-level critical analysis"
    ]
)

# length input
length_input = st.selectbox(
    "Summary Length",[
        "Brief — approximately 300 words",
        "Standard — approximately 700 words",
        "Detailed — approximately 1500 words",
        "Comprehensive — approximately 2500 words"
    ]
)

user_inputs = template.invoke(
     {
          'paper_input':paper_input,
          'style_input':style_input,
          'length_input':length_input
     }
)


if st.button("Summarize"):
     result = model.invoke(user_inputs)
     st.write(result.content[0]['text'])
     
     
# Printint the response
# print(result.content[0]['text'])



