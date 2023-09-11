from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import os
from utils import youtube_transcriptor as trans
import shutil
from dotenv import load_dotenv
import requests_cache
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from typing import Any, Dict, List
import string
import random



load_dotenv()

def run_llm(query: string, chat_history: List[Dict[str, Any]] = []):
    # Embed using OpenAI embeddings
    embeddings =  OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = docsearch.as_retriever(k=5)
    chat_openai = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat_openai, retriever=retriever, return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})

def start_qa(prompt: string):
    button_disabled = len(prompt.strip()) == 0
    st.button("Submit", disabled=button_disabled)
    with st.spinner("Generating response..."):
        if len(prompt) != 0:
            generated_response = run_llm(
                query=prompt, chat_history=st.session_state["chat_history"]
            )

            formatted_response = (
                f"{generated_response['answer']}"
            )

            st.session_state.chat_history.append((prompt, generated_response["answer"]))
            st.session_state.user_prompt_history.append(prompt)
            st.session_state.chat_answers_history.append(formatted_response)

            if st.session_state["chat_answers_history"]:
                for generated_response, user_query in zip(
                    st.session_state["chat_answers_history"],
                    st.session_state["user_prompt_history"],
                ):
                    message(
                        user_query,
                        is_user=True,
                    )
                    res = ''.join(random.choices(string.ascii_letters, k=5))
                    message(generated_response, key=res)

def create_db_and_analye(video_url):
    os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
    if os.path.exists('../db'):
        shutil.rmtree('../db')
    requests_cache.clear()
    transcript=trans.get_transcript(video_url)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(transcript)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(texts,
                                embeddings,
                                metadatas=[{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))],
                                persist_directory="db")
    db.persist()
    st.write("Ready to answer questions.")
    # Question and answering
    prompt = st.text_input("Prompt", placeholder="Enter your question here...")
    if len(prompt) != 0:
        start_qa(prompt)





