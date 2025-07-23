# AI Chat with Video

This project allows users to interact with an AI chatbot that provides answers and generates suggested questions based on the transcript of a YouTube video or uploaded video file. The chatbot uses advanced natural language processing (NLP) techniques to understand the content of the video and generate meaningful responses.

## Features

- **Video Upload**: Upload a video file or provide a YouTube URL.
- **Audio Extraction**: Extract audio from the video and transcribe it to text.
- **AI Chatbot**: Interact with an AI chatbot that answers questions based on the video's content.
- **Suggested Questions**: The chatbot provides suggested questions, which users can click to further explore the video content.
- **Transcript Display**: View the transcript of the video and download it as a text file.
- **Seamless Integration**: The system supports both YouTube URL input and local file uploads.

## Technologies Used

- **Streamlit**: A framework for building interactive applications.
- **yt-dlp**: A tool to download videos from YouTube and other websites.
- **LangChain**: A framework to help with language model interactions. For generating AI responses and suggested questions.
- **FFmpeg**: For extracting audio from video files.
- **Python**: Backend programming language.

## Setup Instructions

### Prerequisites

1. Python 3.7 or above
2. Install dependencies listed in `requirements.txt`.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/iqra-1/AI_chat_with_video.git
   cd AI_chat_with_video
   ```

2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the App

1. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Visit `http://localhost:8501` in your browser to interact with the application.

## How to Use

1. **Upload a Video or Provide a YouTube URL**:
   - Upload a video file (mp4, mov, avi) or paste a YouTube video URL in the sidebar.
   
2. **Process the Video**:
   - Click the "Process Video" button to extract audio from the video, transcribe it, and display the text in the app.
   
3. **Chat with the AI**:
   - Type any question about the video, or click on the suggested questions provided by the chatbot.
   - The AI will respond based on the video's transcript and provide more questions to explore.

4. **Download Transcript**:
   - You can download the video transcript as a `.txt` file.

## Example Use Cases

- **Educational Videos**: Ask the chatbot questions about the content of a lecture or tutorial video.
- **Product Reviews**: Inquire about specific details or reviews discussed in a product review video.
- **Entertainment**: Ask about the storyline, characters, or interesting facts in a movie or TV show video.

## Contributing

Contributions are welcome! If you'd like to help improve this project, please fork the repository and create a pull request with your changes.

### How to Contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes with a clear message.
5. Push your changes to your fork.
6. Open a pull request to the main repository.

## License

This project is licensed under the MIT License.
