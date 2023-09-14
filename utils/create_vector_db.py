import langchain.chains
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

def create_article_from_video(video_url):
    os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
    requests_cache.clear()
    transcript=trans.get_transcript(video_url)
    prompt = """
    Act as an expert copywriter specializing in content optimization for SEO. Your task is to take a given YouTube transcript and transform it into a well-structured and engaging article. Your objectives are as follows:
Content Transformation: Begin by thoroughly reading the provided YouTube transcript. Understand the main ideas, key points, and the overall message conveyed.
Sentence Structure: While rephrasing the content, pay careful attention to sentence structure. Ensure that the article flows logically and coherently.
Keyword Identification: Identify the main keyword or phrase from the transcript. It's crucial to determine the primary topic that the YouTube video discusses.
Keyword Integration: Incorporate the identified keyword naturally throughout the article. Use it in headings, subheadings, and within the body text.
Unique Content: Your goal is to make the article 100% unique. Avoid copying sentences directly from the transcript. Rewrite the content in your own words while retaining the original message and meaning.
SEO Friendliness: Craft the article with SEO best practices in mind. This includes optimizing meta tags (title and meta description), using header tags appropriately, and maintaining an appropriate keyword density.
Engaging and Informative: Ensure that the article is engaging and informative for the reader. It should provide value and insight on the topic discussed in the YouTube video.
By following these guidelines, create a well-optimized, unique, and informative article that would rank well in search engine results and engage readers effectively.
Transcript:{""" + transcript + "}"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(prompt)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    retriever = vectorstore.as_retriever(k=3)
    chat_openai = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat_openai, retriever=retriever, return_source_documents=True
    )
    generated_response = qa({"question": prompt, "chat_history": st.session_state["chat_history"]})
    res2 = ''.join(random.choices(string.ascii_letters, k=5))
    formatted_response = (
        f"{generated_response['answer']}"
    )
    message(formatted_response, key=res2)


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
                        res = ''.join(random.choices(string.ascii_letters, k=5))
                        message(
                            user_query,
                            is_user=True,
                            key=res
                        )
                        res2 = ''.join(random.choices(string.ascii_letters, k=5))
                        message(generated_response, key=res2)





