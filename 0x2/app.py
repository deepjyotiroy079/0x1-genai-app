from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the questions asked."),
        ("user", "Question: {question}")
    ]
)

# streamlit framework
st.title("Langchain Demo with Llama3.1")

input_text = st.text_input("What question do you have in mind?")

# call the llm model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    # provider="auto",  # let Hugging Face choose the best provider for you
)
chat_model = ChatHuggingFace(llm=llm)

output_parser = StrOutputParser()

chain = prompt | chat_model | output_parser

if input_text:
    st.write(chain.invoke({"question": {input_text}}))