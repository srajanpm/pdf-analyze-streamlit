import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any, Dict, List
from streamlit_chat import message
import nest_asyncio
import random
import requests
import string
nest_asyncio.apply()

st.set_page_config(page_title="WebPage Q&A",page_icon="📕")

@st.cache_resource
def load_docs(website_url):
    st.info("`Loading webpage data ...`")
    resp = requests.get(website_url)
    if 200 != resp.status_code:
        return False
    return resp.text

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits


# ...
def run_llm(query: str, chat_history: List[Dict[str, Any]] = [], splits: List[str] = []):
    # Embed using OpenAI embeddings
        embeddings =  OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        vectorstore = FAISS.from_texts(splits, embeddings)

        retriever = vectorstore.as_retriever(k=5)
        chat_openai = ChatOpenAI(verbose=True, temperature=0)

        qa = ConversationalRetrievalChain.from_llm(
            llm=chat_openai, retriever=retriever, return_source_documents=True
        )

        return qa({"question": query, "chat_history": chat_history})

def main():
    if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
    ):
        st.session_state["chat_answers_history"] = []
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_history"] = []

    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
        <p>Made by <a href='https://github.com/hbahuguna/'>Himanshu Bahuguna</a></p>
        <p><a href='https://yep.so/p/omnichat'>Join OmniChat Wait list</a></p>
    </div>
    """

    st.markdown(foot, unsafe_allow_html=True)

    # Add custom CSS
    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }
            
           
       
            
          

        </style>
        """,
        unsafe_allow_html=True,
    )
    st.text("")
    st.text("")
    st.text("")





    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <p><h1 style="display: inline-block;">Question & Answer with</h1></p>
    </div>
    <div style="display: flex; align-items: center; margin-left: 0;">
    <p><h1 style="display: inline-block;">Webpage</h1></p>
    </div>
    """,
    unsafe_allow_html=True,
        )






    st.sidebar.title("Menu")

    embedding_option = st.sidebar.radio(
        "Embeddings", ["OpenAI Embeddings"])



    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    from PIL import Image
    image = Image.open('img/logo1.png')

    st.sidebar.image(image)

    website_url = st.text_input("Please enter Webpage URL")
    button_disabled = len(website_url) == 0
    st.button("Submit Website", disabled=button_disabled)
    if len(website_url) != 0:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=0,
        )
        loaded_text = load_docs(website_url)

        # Split the document into chunks
        splitter_type = "RecursiveCharacterTextSplitter"
        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=0, split_method=splitter_type)
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        vectorstore = FAISS.from_texts(splits, embeddings)
        retriever = vectorstore.as_retriever(k=5)
        chat_openai = ChatOpenAI(verbose=True, temperature=0)

        qa = ConversationalRetrievalChain.from_llm(
            llm=chat_openai, retriever=retriever, return_source_documents=True
        )
        st.write("Ready to answer questions.")
        # Question and answering
        prompt = st.text_input("Prompt", placeholder="Enter your question here...")
        if len(prompt) != 0:
            res1 = ''.join(random.choices(string.ascii_letters, k=7))
            st.button("Submit", key=res1)
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
                            res2 = ''.join(random.choices(string.ascii_letters, k=5))
                            message(generated_response, key=res2)


if __name__ == "__main__":
    main()
