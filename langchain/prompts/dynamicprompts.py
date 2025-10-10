# Dynamic prompts from the user using Langchain
# Author: Muhammad Humayun Khan

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

model = ChatOpenAI()

# the three placeholder in the UI as paper_input, style_input and length_input
paper_input = st.selectbox("Select Research Paper Name",["Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers","GPT-3: Language Models are Few-Shot Learners","Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation style",["Beginner-friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraph)","Medium (3-5 paragraphs)","Long (detailed explanation)"])

# load the template which was created by using the template_generator.py
template = load_prompt('template.json')

# fill the placeholders(declared above) with the input_variables and we will get the user prompt
prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
}) 

# now user prompt to the model
if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)
