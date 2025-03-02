import streamlit as st
import yt_dlp
import tempfile
import os

from extract_text import extract_audio_from_video, transcribe_audio
from process_text import process_extracted_text
from langchain_integration import generate_answer_and_suggested_questions

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def sanitize_filename(filename):
    """Clean filename by replacing spaces and removing invalid characters"""
    return "".join(
        [
            c if c.isalnum() or c in ("_", "-", ".") else "_"
            for c in filename.replace(" ", "_")
        ]
    )


def download_youtube_video(url, file_format="mp4"):
    """Download a YouTube video using yt-dlp."""
    ydl_opts = {
        "format": f"bestvideo[ext={file_format}]+bestaudio[ext=m4a]/best[ext={file_format}]",
        "outtmpl": "% (title)s.%(ext)s",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return {
                "title": info["title"],
                "file_path": f"{info['title']}.{file_format}",
            }
    except Exception as e:
        st.error(f"Failed to download video: {str(e)}")
        return None


# Streamlit UI
st.set_page_config(page_title="YouTube AI Chatbot", layout="wide")
st.title("YouTube Video Analyzer & AI Chatbot")

# Initialize session state
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "processed_text" not in st.session_state:
    st.session_state.processed_text = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
st.sidebar.header("Upload or Paste URL")
input_method = st.sidebar.radio("Choose input method:", ("YouTube URL", "File Upload"))

video_source, youtube_url, uploaded_file = None, None, None
if input_method == "YouTube URL":
    youtube_url = st.sidebar.text_input("Enter YouTube URL:")
    if youtube_url:
        video_source = "youtube"
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video file:", type=["mp4", "mov", "avi"]
    )
    if uploaded_file:
        video_source = "upload"

# Video Processing
if st.sidebar.button("Process Video"):
    if not video_source:
        st.warning("Please provide a video source!")
    else:
        # Delete previous video if exists
        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
            os.remove(st.session_state.video_path)
            st.session_state.video_path = None

        with st.spinner("Processing video..."):
            if video_source == "youtube":
                video_info = download_youtube_video(youtube_url)
                if video_info is None:
                    st.stop()
                video_path = video_info["file_path"]
            else:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    video_path = tmp_file.name

            st.session_state.video_path = video_path
            audio_file = extract_audio_from_video(video_path)
            transcription = transcribe_audio(audio_file)
            st.session_state.processed_text = process_extracted_text(transcription)

            # Clean up temp files
            os.remove(audio_file)

            st.success("Processing complete!")

# Display Video
if st.session_state.video_path:
    st.video(st.session_state.video_path)

# Show Script
if st.session_state.processed_text:
    with st.expander("Show Transcript"):
        st.text_area("Transcript", st.session_state.processed_text, height=200)
    st.download_button(
        "Download Transcript",
        data=st.session_state.processed_text,
        file_name="transcript.txt",
        mime="text/plain",
    )

# Initialize session state variables
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "processed_text" not in st.session_state:
    st.session_state.processed_text = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Chatbot UI
if st.session_state.processed_text:
    st.subheader("Chat with AI about the Video")

    chat_container = st.container()
    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"**{chat['sender'].capitalize()}:** {chat['message']}")
            if chat.get("suggestions"):
                for j, suggestion in enumerate(chat["suggestions"]):
                    if st.button(
                        suggestion, key=f"suggestion_{i}_{j}_{hash(suggestion)}"
                    ):
                        st.session_state.pending_question = suggestion
                        st.rerun()  # Ensure the clicked question is processed correctly

# Fixed bottom input field
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    prompt = st.chat_input("Say something")

if prompt:
    try:
        with st.spinner("Generating response..."):
            result = generate_answer_and_suggested_questions(
                prompt, st.session_state.processed_text
            )
            response = result["answer"]
            suggested_questions = result["suggested_questions"]

        # Update chat history
        st.session_state.chat_history.append({"sender": "user", "message": prompt})
        st.session_state.chat_history.append(
            {"sender": "bot", "message": response, "suggestions": suggested_questions}
        )

        # Rerun to refresh UI
        st.rerun()

    except Exception as e:
        st.error(f"Chatbot error: {str(e)}")
