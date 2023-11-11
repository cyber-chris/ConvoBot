import torch
from TTS.api import TTS
import subprocess

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: need to find a good voice
output_path = "../output/speech_test.wav"
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True).to(device)
for speaker in ['p236', 'p286', '317'] + tts.speakers[80:]:
	print(speaker)
	tts.tts_to_file(text="Hey, Jarvis here, how can I help today?", file_path=output_path, speaker=speaker)
	subprocess.run(['aplay', output_path], check=True)