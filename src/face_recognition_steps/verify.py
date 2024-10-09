import pickle
from deepface.modules import verification
from deepface import DeepFace
import cv2
from tqdm import tqdm

VECTOR_DB_PATH = "vector_db.pkl"
vector_db = pickle.load(open(VECTOR_DB_PATH, "rb"))

embeddings = [emb[0] for emb in vector_db]
names = [emb[1] for emb in vector_db]

unkown_person_img = cv2.imread("images/David-Gilmour.jpg")

unkown_person_rep = DeepFace.represent(
    unkown_person_img,
    detector_backend="yunet",
    model_name="Facenet",
    align=True
)

unkown_person_emb = unkown_person_rep[0]["embedding"]

print("Calculando distância euclidiana entre os embeddings...")
min_distance = float('inf') 
print(min_distance)
min_distance_indentity = ""

for emb, name in tqdm(zip(embeddings, names), total=len(embeddings)):
    distance = verification.find_euclidean_distance(unkown_person_emb, emb)
    identity = name.replace('_', ' ').title()

    print(f"d(\"Pessoa desconhecida\", \"{identity}\")","=", f"{distance}")

    if distance < min_distance:
        min_distance = distance
        min_distance_identity = identity

print("Identidade encontrada:", min_distance_identity)
print("Distância calculada:", min_distance)
