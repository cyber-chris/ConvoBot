from ctransformers import AutoModelForCausalLM
import time
import os

model_path = os.getenv("MODEL_PATH") or "/home/ct/llm-models/llama-2-13b-chat.Q4_K_M.gguf"

print(f"Loading {model_path}")
preload = time.time()

# I have too little GPU VRAM to actually use my GPU, I think.
llm = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama", gpu_layers=0)

postload = time.time()
print(f"Loaded in {postload - preload}")

personality = "Personality: You are Jarvis, a helpful assistant of mine."

for x in llm("What are the standard model formats used for pre-trained GPT models?", stream=True):
    print(x, end='')
print('')

while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break

    for output in llm(f"{personality}\nUser: {user_input}", stream=True):
            pass