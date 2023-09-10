from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(youtube_url):
    video_id = YouTube(youtube_url).video_id

    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    text_transcript = ""

    for segment in transcript:
        text_transcript += segment['text'] + " "

    return text_transcript
