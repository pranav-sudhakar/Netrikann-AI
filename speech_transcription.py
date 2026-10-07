import speech_recognition as sr
import os

def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    global return_item
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = r'keys/blindexa-ai.json'
    try:
        print("Google Cloud Speech thinks you said " + r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS))
        return_item = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
        return str(return_item)
    except sr.UnknownValueError:
        print("Google Cloud Speech could not understand audio")
        return False
    except sr.RequestError as e:
        print("Could not request results from Google Cloud Speech service; {0}".format(e))
        return False