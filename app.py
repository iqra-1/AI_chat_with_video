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
    sanitized = filename.replace(" ", "_")
    sanitized = "".join(
        [c if c.isalnum() or c in ("_", "-", ".") else "_" for c in sanitized]
    )
    return sanitized


def download_youtube_video(url, file_format="mp4"):
    """
    Download a YouTube video using yt-dlp.

    Args:
        url (str): YouTube video URL.
        file_format (str): Desired file format for the video (default is 'mp4').

    Returns:
        dict: Video metadata including title and downloaded file path.
    """
    ydl_opts = {
        "format": f"bestvideo[ext={file_format}]+bestaudio[ext=m4a]/best[ext={file_format}]",
        "outtmpl": "%(title)s.%(ext)s",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = f"{info['title']}.{file_format}"
        return {
            "title": info["title"],
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
        }


# Streamlit UI
st.title("YouTube Video Analyzer & AI Question Generator")

# Sidebar configuration
st.sidebar.header("Input Configuration")
input_method = st.sidebar.radio("Select input method:", ("YouTube URL", "File Upload"))

video_source = None
youtube_url = None
uploaded_file = None

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

# Main content area
if video_source == "youtube":
    st.header("YouTube Video Details")
    st.subheader("Selected URL:")
    st.write(youtube_url)

elif video_source == "upload":
    st.header("Uploaded File Details")
    st.subheader("Selected File:")
    st.write(uploaded_file.name)

# Processing section
st.header("Video Analysis & Question Generation")
if st.button("Process Video"):
    if video_source is None:
        st.warning("Please select a video source first!")
        st.stop()

    with st.spinner("Analyzing video content..."):
        try:
            # Handle different input sources
            if video_source == "youtube":
                if (
                    "video_info" not in st.session_state
                    or st.session_state.video_info.get("url") != youtube_url
                ):
                    with st.spinner("Downloading video..."):
                        video_info = download_youtube_video(youtube_url)
                        video_info["url"] = youtube_url  # Store URL for validation
                        st.session_state.video_info = video_info
                video_path = st.session_state.video_info["file_path"]

                # Display the downloaded video
                st.subheader("Video Preview")
                st.video(video_path)
            else:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    video_path = tmp_file.name

                # Display the uploaded video
                st.subheader("Video Preview")
                st.video(video_path)

            # Audio extraction
            st.divider()
            with st.spinner("Extracting audio..."):
                audio_file = extract_audio_from_video(video_path)
                st.success("Audio extracted successfully")

            # Transcription
            st.divider()
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(audio_file)
                st.subheader("Transcription:")
                st.text_area(
                    "Full Transcription",
                    transcription,
                    height=200,
                    label_visibility="collapsed",
                )

            # Text processing
            st.divider()
            with st.spinner("Processing text..."):
                processed_text = process_extracted_text(transcription)
                st.session_state.processed_text = (
                    processed_text  # Store in session state
                )
                st.subheader("Processed Text:")
                st.text_area(
                    "Important Content",
                    processed_text,
                    height=200,
                    label_visibility="collapsed",
                )

                # Generate download filename
                if video_source == "youtube":
                    raw_filename = (
                        f"{st.session_state.video_info['title']}_transcript.txt"
                    )
                else:
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    raw_filename = f"{base_name}_transcript.txt"

                filename = sanitize_filename(raw_filename)

                # Add download button
                st.download_button(
                    label="Download Processed Transcript",
                    data=processed_text,
                    file_name=filename,
                    mime="text/plain",
                )

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
        finally:
            # Clean up temporary files for uploaded videos
            if video_source == "upload" and "video_path" in locals():
                try:
                    os.unlink(video_path)
                except Exception as e:
                    st.warning(f"Could not clean up temporary file: {str(e)}")

# Question handling – Chatbot UI
if "processed_text" in st.session_state:

    # Initialize chat history if not already in session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Check if a suggested question was clicked and saved as pending
    if st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = ""  # clear the pending question
        st.session_state.chat_history.append({"sender": "user", "message": question})
        with st.spinner("Generating answer..."):
            response = generate_answer_and_suggested_questions(
                st.session_state.processed_text, question
            )
            bot_message = {
                "sender": "bot",
                "message": response["answer"],
                "suggestions": response["suggested_questions"],
            }
            st.session_state.chat_history.append(bot_message)
        st.rerun()  # rerun to update the chat interface

    st.divider()
    st.markdown("## Chatbot")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg_index, chat in enumerate(st.session_state.chat_history):
            if chat["sender"] == "user":
                st.markdown(f"**User:** {chat['message']}")
            else:
                st.markdown(f"**Bot:** {chat['message']}")
                # Display suggested follow-up questions as clickable buttons
                if chat.get("suggestions"):
                    cols = st.columns(len(chat["suggestions"]))
                    for idx, suggestion in enumerate(chat["suggestions"]):
                        # Incorporate both the chat message index and suggestion index for uniqueness
                        if cols[idx].button(suggestion, key=f"sugg_{msg_index}_{idx}"):
                            st.session_state.pending_question = suggestion
                            st.rerun()

    # Input form at the bottom so user can type the next question
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_input("Enter your question about the video content:")
        submitted = st.form_submit_button("Send")
        if submitted and user_question:
            st.session_state.chat_history.append(
                {"sender": "user", "message": user_question}
            )
            with st.spinner("Generating answer..."):
                response = generate_answer_and_suggested_questions(
                    st.session_state.processed_text, user_question
                )
                bot_message = {
                    "sender": "bot",
                    "message": response["answer"],
                    "suggestions": response["suggested_questions"],
                }
                st.session_state.chat_history.append(bot_message)
            st.rerun()  # update the conversation immediately

    # Allow user to view the transcript if desired
    with st.expander("Show Transcript"):
        st.text_area("Full Transcription", st.session_state.processed_text, height=200)
