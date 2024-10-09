from contextlib import asynccontextmanager
import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import Image, Person

from deepface import DeepFace
from deepface.modules import verification
from tqdm import tqdm
import pickle

def generate_data_url(base64: str):
    return f'data:image/jpg;base64,{base64}'


#@asynccontextmanager
#async def lifespan(app: FastAPI):
#    # Carregar modelos...
#    yield
#    # Limpar recursos/modelos

VECTOR_DB_PATH = "vector_db.pkl"
vector_db = pickle.load(open(VECTOR_DB_PATH, "rb"))

app = FastAPI(root_path="/deteccao-facial-api")
#origins = ["http://localhost:8001", "http://localhost:4200", "http://localhost:8080"]
#
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=origins,
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)


@app.get("/status")
def status() -> dict:
    return {"status": "ok"}

@app.post("/echo")
def echo(data: dict) -> dict:
    return data

@app.post("/detect-face")
def detect_face(image: Image) -> list:
    try:
        facial_areas = []
        faces = DeepFace.extract_faces(
            generate_data_url(image.base64),
            detector_backend="yunet",
            align=True
        )
        for face in faces:
            facial_areas.append({"facial_area": face["facial_area"]})
        return facial_areas 
    except Exception as e:
        print("Erro:", e)
        return []
    return [] 

@app.post("/recognize-face")
def recognize_face(image: Image) -> Person:
    return Person(name="Fulano de tal", distance=42)
