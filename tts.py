import os
import audio_player
from google.cloud import texttospeech_v1

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'keys/blindexa-ai.json'

client = texttospeech_v1.TextToSpeechClient()

def speak(text):
    synthesis_input = texttospeech_v1.SynthesisInput(text=text)

    voice = texttospeech_v1.VoiceSelectionParams(
    language_code='en-US',   
    name='en-US-Studio-O')

    audio_config = texttospeech_v1.AudioConfig(
    audio_encoding = texttospeech_v1.AudioEncoding.MP3,
    effects_profile_id = ['small-bluetooth-speaker-class-device'],
    speaking_rate = 1,
    pitch = 1)

    response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config)

    filename = "temp_audio_file.mp3"
    with open(filename,'wb') as output:
        output.write(response.audio_content)
    
    audio_player.play_clip(filename)
    os.remove(filename)