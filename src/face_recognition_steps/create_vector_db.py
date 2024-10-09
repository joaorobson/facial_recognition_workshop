import os
import pickle

from deepface import DeepFace
from tqdm import tqdm


FACES_DIR = "faces"
REPRESENTATION_MODEL = "Facenet"
DETECTION_MODEL = "yunet"
VECTOR_DB_PATH = "vector_db.pkl"

def get_images_paths():
    people_dirs = os.listdir(FACES_DIR)
    images = []

    for person_dir in people_dirs:
        for image_path in os.listdir(os.path.join(FACES_DIR, person_dir)):
            images.append((os.path.join(FACES_DIR, person_dir, image_path), person_dir))

    return images

def generate_embeddings(images):
    embeddings = []

    for image in tqdm(images):
        image_path, person = image
        try:
            embs = DeepFace.represent( 
                image_path,
                detector_backend=DETECTION_MODEL,
                model_name=REPRESENTATION_MODEL,
                align=True
            )
        except Exception as e:
            print("Erro", image_path)
            embs = []
        
        if len(embs) == 1:
            embeddings.append((embs[0]["embedding"], person))

    return embeddings

def store_embeddings(embeddings):
    pickle.dump(embeddings, open(VECTOR_DB_PATH, "wb"))

images = get_images_paths()
embeddings = generate_embeddings(images)
store_embeddings(embeddings)
print("Vetores salvos em:", VECTOR_DB_PATH)
