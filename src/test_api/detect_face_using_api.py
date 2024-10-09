import requests

import base64
import json

# Função para converter imagem para base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Exemplo de uso
image_path = "images/David-Gilmour.jpg"
base64_string = image_to_base64(image_path)

# Imprime a string base64
print(base64_string[:10])

API_URL = "http://127.0.0.1:8000/detect-face"
resp = requests.post(API_URL, data=json.dumps({"base64": base64_string}))
print(resp.json())
