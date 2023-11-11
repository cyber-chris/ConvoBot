from typing import Optional
from ctransformers import AutoModelForCausalLM
import time
import os
import pyttsx3
import string
import speech_recognition as sr

USER_SPEAK=True

def load_model():
    model_path = os.getenv("MODEL_PATH") or "/home/ct/llm-models/llama-2-13b-chat.Q4_K_M.gguf"

    print(f"Loading {model_path}")
    preload = time.time()

    # I have too little GPU VRAM to actually use my GPU, I think.
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, model_type="llama", gpu_layers=0)

    postload = time.time()
    print(f"Loaded in {postload - preload}")

    return llm

def contains_punctuation(text) -> bool:
    return any(c in string.punctuation for c in reversed(text))

def speech_to_text(recognizer) -> str:
    with sr.Microphone() as source:
        print("User: ", end='', flush=True)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_whisper(audio, model="base.en", language="english")
    except:
        return ''

def jarvis_speak(text):
    pyttsx3.speak(text)

llm = load_model()
r = sr.Recognizer() if USER_SPEAK else None

personality = "Personality: You are Jarvis, an intelligent and helpful assistant of mine."

while True:
    user_input = ''
    if USER_SPEAK:
        user_input = speech_to_text(r)
        print(user_input)
    else:
        user_input = input("User: ")
    
    if not user_input:
        apology = "Apologies, I didn't catch that."
        print(apology)
        jarvis_speak(apology)
        continue

	# I should evaluate the generator in a separate process? Use some multiprocessing.
    speech_buffer = []
    for output in llm(f"{personality}\nUser: {user_input}\nResponse:", stream=True, reset=False):
        print(output, end='', flush=True)
        speech_buffer.append(output)
        if contains_punctuation(output):
            jarvis_speak(''.join(speech_buffer))
            speech_buffer = []
    print()
