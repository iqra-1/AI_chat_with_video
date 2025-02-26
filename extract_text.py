import whisper  # Whisper model for transcribing audio
from moviepy.video.io.VideoFileClip import VideoFileClip


def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".wav")  # Change extension to .wav
    video.audio.write_audiofile(audio_path)
    print(f"Extracted audio saved to: {audio_path}")  # Debugging line
    return audio_path


def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # You can use a different model size if desired
    result = model.transcribe(audio_path)
    return result["text"]
