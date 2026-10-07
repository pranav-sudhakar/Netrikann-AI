import os, io
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'keys/blindexa-ai.json'

client = vision_v1.ImageAnnotatorClient()

def scrutinize_image(filename):
    with io.open(filename, 'rb') as image_file:
        content = image_file.read()

    image = vision_v1.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    text_data = []
    for text in texts:
        text_dict = {
            'locale': text.locale,
            'description': text.description
        }
        text_data.append(text_dict)

    df = pd.DataFrame(text_data)
    raw_data = list(df['description'].iloc[0])
    # Print the first description
    result = ""
    for char in raw_data:
        if char == "\n":
            result = result + " "
        else:
            result = result + char

    return str(result)