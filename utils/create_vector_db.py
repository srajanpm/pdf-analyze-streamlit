from langchain.embeddings.openai import OpenAIEmbeddings
import os
from utils import youtube_transcriptor as trans
from dotenv import load_dotenv
import requests_cache
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
import string
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()

def create_db_and_analye(video_url):
    os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
    requests_cache.clear()
    transcript=trans.get_transcript(video_url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(transcript)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    retriever = vectorstore.as_retriever(k=3)
    chat_openai = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat_openai, retriever=retriever, return_source_documents=True
    )
    st.write("Ready to answer questions.")
    # Question and answering
    prompt = st.text_input("Prompt", placeholder="Enter your question here...")
    if len(prompt) != 0:
        st.button("Submit")
        with st.spinner("Generating response..."):
            if len(prompt) != 0:
                generated_response = qa({"question": prompt, "chat_history": st.session_state["chat_history"]})

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





