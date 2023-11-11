import pyttsx3

engine = pyttsx3.init()
for voice in ['english', 'english_rp', 'english-us', 'english-north']:
   engine.setProperty('voice', voice)
   engine.say('The quick brown fox jumped over the lazy dog.')
engine.runAndWait()