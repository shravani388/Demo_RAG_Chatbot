import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_community.llms import Ollama

st.set_page_config(page_title="RAG Chatbot")
st.title("RAG Chatbot")
st.write("Please enter your question")

@st.cache_resource
def loadvector():
    loader = TextLoader('c++_Introduction.txt', encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedding)
    return db

db = loadvector()

llm = Ollama(model = "gemma2:2b")

user_question = st.text_input("Ask your question about C++")
if user_question:
    with st.spinner("Thinking...."):
        docs = db.similarity_search(user_question)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using the context below

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        response = llm.invoke(prompt)

    st.subheader("Answer: ")
    st.write(response)