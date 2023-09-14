import os
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
from pytube import YouTube
from utils import create_vector_db as mydb
from dotenv import load_dotenv

load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]

def main():
    st.set_page_config(page_title="Youtube Q&A",page_icon=":yt:")
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
        <p><h1 style="display: inline-block;">Create Article From</h1></p>
    </div>
    <div style="display: flex; align-items: center; margin-left: 0;">
    <p><h1 style="display: inline-block;">Youtube Video</h1></p>
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

    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"
    video_url = st.text_input("Video URL")
    button_disabled = len(video_url) == 0
    st.button("Submit Video", disabled=button_disabled)
    if len(video_url) != 0:
        analyze_video(video_url)

def analyze_video(video_url: str):
    st.text("")
    st.text("")
    st.text("")
    video_id = YouTube(video_url).video_id
    # Construct the URL of the thumbnail image
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

    # Download the thumbnail image and convert it to a PIL Image object
    response = requests.get(thumbnail_url)
    img = Image.open(BytesIO(response.content))

    # Display the thumbnail image in Streamlit
    st.image(img)
    mydb.create_article_from_video(video_url)



if __name__ == "__main__":
    main()
