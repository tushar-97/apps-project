import os
import subprocess

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from transformers import pipeline

from flask import Flask

VIDEO_FOLDER = "videos"
AUDIO_FOLDER = "audios"

app = Flask(__name__)

speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-base")


def get_video(video_name: str):
    account_url = "https://cs6211sys.blob.core.windows.net"
    default_credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(account_url, credential=default_credential)

    container_client = blob_service_client.get_container_client(container="videos")

    blob_list = container_client.list_blob_names()
    blob_names = []
    for blob in blob_list:
        blob_names.append(blob)

    assert video_name in blob_names, f"{video_name} has not been uploaded to blob storage"

    if not os.path.isfile(os.path.join(VIDEO_FOLDER, video_name)):
        with open(os.path.join(VIDEO_FOLDER, video_name), "wb") as download_file:
            download_file.write(container_client.download_blob(video_name).readall())


def extract_audio(video_name: str):
    input_path = os.path.join(VIDEO_FOLDER, video_name)
    output_path = os.path.join(AUDIO_FOLDER, get_audio_file_name(video_name))
    subprocess.call(['ffmpeg', '-y', '-i', input_path, output_path])


def transcribe_video(video_name: str):
    file = os.path.join(AUDIO_FOLDER, get_audio_file_name(video_name))
    result = speech_to_text(file)
    transcript = result['text']

    return transcript


def get_audio_file_name(video_name: str):
    base_name, _ = os.path.splitext(video_name)
    return f"{base_name}.wav"


@app.route('/')
def index():
    return "Transcript Server"


@app.route('/submit/<video_name>')
def submit(video_name):
    get_video(video_name)
    extract_audio(video_name)
    return transcribe_video(video_name)


if __name__ == "__main__":
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    # app.run()
