import json
import requests

API_URL = "http://127.0.0.1:8000/status"

resp = requests.get(API_URL)

print("Resultado do GET em /status:")
print(resp.json())

API_URL = "http://127.0.0.1:8000/echo"

resp = requests.post(API_URL, data=json.dumps({"ola": "mundo"}))

print("Resultado do POST em /echo:")
print(resp.json())

