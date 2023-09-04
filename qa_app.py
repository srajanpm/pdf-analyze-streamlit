import os
import PyPDF2
import streamlit as st
from io import StringIO
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any, Dict, List
from streamlit_chat import message


st.set_page_config(page_title="PDF Q&A",page_icon="📕")


loaded_text=""
@st.cache_data
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text


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
    st.sidebar.image("img/logo1.png")


   

    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">Question & Answer with</h1>
        <h1 style="display: inline-block;">PDF & Txt Files</h1>
    </div>
    """,
    unsafe_allow_html=True,
        )
    
    


    
    
    st.sidebar.title("Menu")
    
    embedding_option = st.sidebar.radio(
        "Embeddings", ["OpenAI Embeddings"])

    
    retriever_type = st.sidebar.selectbox(
        "Retriever", ["SIMILARITY SEARCH"])

    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=50, split_method=splitter_type)


        st.write("Ready to answer questions.")

        # Question and answering
        prompt = st.text_input("Prompt", placeholder="Enter your question here...") or st.button(
            "Submit"
        )
        if prompt:
            with st.spinner("Generating response..."):
                generated_response = run_llm(
                    query=prompt, chat_history=st.session_state["chat_history"], splits = splits
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
                message(generated_response)


if __name__ == "__main__":
    main()
