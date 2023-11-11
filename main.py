from ctransformers import AutoModelForCausalLM
import time
import os
import pyttsx3
import string

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

llm = load_model()

personality = "Personality: You are Jarvis, an intelligent and helpful assistant of mine."

while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break

	# I should evaluate the generator in a separate process? Use some multiprocessing.
    speech_buffer = []
    for output in llm(f"{personality}\nUser: {user_input}\nResponse:", stream=True, reset=False):
        print(output, end='', flush=True)
        speech_buffer.append(output)
        if contains_punctuation(output):
            pyttsx3.speak(''.join(speech_buffer))
            speech_buffer = []
    print()
