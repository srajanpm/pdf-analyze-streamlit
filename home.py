import streamlit as st


st.set_page_config(page_title='Chat with multiple Data Sources️‍', page_icon='📕')


body = f"""
<div style="
    top: 10;
    text-align: left;
">
    <p><a href='/chat_with_pdf_txt_file'>Chat With PDF & TXT Files</a></p>
    <p><a href='/chat_with_website'>Chat With Webpage </a></p>
    <p><a href='/youtube_transcript_to_article'>Create Article from Youtube Video</a></p>
    <p><a href='/chat_with_youtube_videos'>Chat With Youtube Video</a></p>
    <p><a href='/chat_with_website'>Chat With Website sitemap xml Data</a></p>
</div>
"""

st.markdown("<h1 style='text-align: center; color: red;'>OmniChat</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: left; color: red;'>App to Chat with Many Data Sources</h1>", unsafe_allow_html=True)

st.markdown("\n")
st.markdown("\n")
st.markdown(body, unsafe_allow_html=True)
st.markdown("\n")
st.markdown("\n")
from PIL import Image
image = Image.open('img/logo1.png')

st.image(image)
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
foot = f"""
<div style="
    bottom: 0;
    text-align: center;
">
    <p>Made by <a href='https://github.com/hbahuguna/'>Himanshu Bahuguna</a></p>
    <p><a href='https://yep.so/p/omnichat'>Join OmniChat Wait list</a></p>
</div>
"""

st.markdown(foot, unsafe_allow_html=True)

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
from PIL import Image
image = Image.open('img/logo1.png')

st.sidebar.image(image)

