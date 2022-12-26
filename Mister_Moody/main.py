import pyaudio
import wave
import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import schedule
import time
from dotenv import load_dotenv
from hue import Bridge, Light
import asyncio

load_dotenv()
r = sr.Recognizer()
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

light = Light(4, ip="192.168.1.200", user=os.getenv("HUE_USERNAME"))

try:
    asyncio.run(light.set_state({"on": True, "sat": 10, "bri": 64, "hue": 10000}))
except Exception as e:
    print(e)
    asyncio.run(Bridge.discover())
    light = Light(4, ip="192.168.1.200", user=os.getenv("HUE_USERNAME"))
    asyncio.run(light.set_state({"on": True, "sat": 10, "bri": 64, "hue": 10000}))


def record_snippet(out_file="output.wav", record_time=10):
    chunk = 1024
    audio_format = pyaudio.paInt16
    channels = 2
    rate = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("* recording")

    frames = []

    for i in range(0, int(rate / chunk * record_time)):
        data = stream.read(chunk)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(out_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def simple_stt(infile="output.wav"):
    with sr.AudioFile(infile) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        print(text)


def get_large_audio_transcription(path):
    sound = AudioSegment.from_wav(path)
    chunks = split_on_silence(sound,
                              min_silence_len=500,
                              silence_thresh=sound.dBFS-14,
                              keep_silence=500,)

    folder_name = "audio-chunks"

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""

    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        print(chunk_filename)
        audio_chunk.export(chunk_filename, format="wav")
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", e)
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    return whole_text


def simple_sentiment(text=""):
    return sia.polarity_scores(text)


def score_to_color(score):
    print(score['compound'])
    if score['compound'] < -0.1:
        asyncio.run(light.set_state({"sat": 254, "bri": 254, "hue": 45000}))

    elif score['compound'] > 0.1:
        asyncio.run(light.set_state({"sat": 254, "bri": 254, "hue": 9000}))

    else:
        asyncio.run(light.set_state({"sat": 10, "bri": 64, "hue": 10000}))


def job():
    record_snippet()
    text = get_large_audio_transcription("output.wav")
    scores = simple_sentiment(text)
    score_to_color(scores)


if __name__ == '__main__':

    schedule.every(1).seconds.do(job)
    total_time = 0

    while True and total_time <= 120:
        schedule.run_pending()
        time.sleep(1)
        total_time += 10
