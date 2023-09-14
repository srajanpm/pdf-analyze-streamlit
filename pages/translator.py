import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
import string
import random

load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]

def main():
    st.set_page_config(page_title="Translator",page_icon=":yt:")
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
        <p><h1 style="display: inline-block;">Translate </h1></p>
    </div>
    <div style="display: flex; align-items: center; margin-left: 0;">
    <p><h1 style="display: inline-block;">From English to Any Language</h1></p>
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

    output_lang = st.text_input("Output Language")
    input_text = st.text_area("Please provide the text to translate")
    st.button("Translate Text")
    if len(output_lang) != 0 and len(input_text) != 0:
        prompt = "You are a translator with vast knowledge of human languages. Please translate {" + input_text + "} from English  to {"+output_lang+"}"
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(prompt)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts, embeddings)
        retriever = vectorstore.as_retriever(k=3)
        chat_openai = ChatOpenAI(verbose=True, temperature=0)
        qa = ConversationalRetrievalChain.from_llm(
            llm=chat_openai, retriever=retriever, return_source_documents=True
        )
        st.write("Ready to translate text.")
        with st.spinner("Translating text..."):
            generated_response = qa({"question": prompt, "chat_history": st.session_state["chat_history"]})
            res2 = ''.join(random.choices(string.ascii_letters, k=5))
            formatted_response = (
                f"{generated_response['answer']}"
            )
            message(formatted_response, key=res2)




if __name__ == "__main__":
    main()
