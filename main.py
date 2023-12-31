from typing import Optional
from ctransformers import AutoModelForCausalLM
import time
import os
import pyttsx3
import string
import speech_recognition as sr
import torch
from TTS.api import TTS
import subprocess
import threading
from nltk.tokenize import sent_tokenize
from collections import deque
import sounddevice as sd
import numpy as np

USER_SPEAK = True
JARVIS_SPEAK = True


def load_model():
    model_path = (
        os.getenv("MODEL_PATH") or "/home/ct/llm-models/llama-2-13b-chat.Q4_K_M.gguf"
    )

    print(f"Loading {model_path}")
    preload = time.time()

    # I have too little GPU VRAM to actually use my GPU, I think.
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, model_type="llama", max_new_tokens=512, gpu_layers=0, context_length=1024
    )

    postload = time.time()
    print(f"Loaded in {postload - preload}")

    return llm


def contains_punctuation(text) -> bool:
    return any(c in string.punctuation for c in reversed(text))


def is_sentence(text) -> bool:
    tks = sent_tokenize(text)
    return tks and (len(tks) > 1 or tks[-1][-1] in (',', '.', '!', '?', ':', ';'))


def speech_to_text() -> str:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("User: ", end="", flush=True)
        audio = r.listen(source)
    try:
        return r.recognize_whisper(audio, model="base.en", language="english")
    except:
        return ""


def jarvis_speak(tts, text, deque=None):
    if not JARVIS_SPEAK or not text.strip():
        return

    wav = tts.tts(text=text, speaker="p236")

    fs = 22050
    audio_data = np.array(wav)
    scaled = (audio_data * 1.4).clip(-1, 1)
    sd.play(scaled, fs, blocking=True, blocksize=2048)


def queue_worker(tts, deque, event):
    buffer = ""

    while True:
        if not deque:
            event.wait()
        event.clear()
        val = deque.popleft()
        if val is None:
            break

        buffer += val
        if is_sentence(buffer):
            jarvis_speak(tts, buffer)
            buffer = ""
    if buffer:
        jarvis_speak(tts, buffer)


def completion(text):
    available = threading.Event()
    q = deque()

    worker = threading.Thread(target=queue_worker, args=(tts, q, available))
    worker.start()

    for output in llm(text, stream=True, reset=False):
        print(output, end="", flush=True)
        q.append(output)
        available.set()
    print()
    q.append(None)
    available.set()
    worker.join()


llm = load_model()
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False).to("cuda")

personality = (
    "Personality: You are Jarvis, an intelligent and helpful assistant to me, Chris."
)

while True:
    user_input = ""
    if USER_SPEAK:
        user_input = speech_to_text()
        print(user_input)
    else:
        user_input = input("Chris: ")

    if not user_input:
        apology = "Apologies, I didn't catch that."
        print(apology)
        jarvis_speak(tts, apology)
        continue

    text = f"{personality}\nChris: {user_input}\nResponse:"

    completion(text)
