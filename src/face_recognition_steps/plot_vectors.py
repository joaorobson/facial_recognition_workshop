import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import seaborn as sns
import sys
from deepface.modules import verification
from deepface import DeepFace
import cv2

VECTOR_DB_PATH = "vector_db.pkl"
vector_db = pickle.load(open(VECTOR_DB_PATH, "rb"))

embeddings = [emb[0] for emb in vector_db]
names = [emb[1] for emb in vector_db]

for e in vector_db:
    print(f'[{e[0][0]:.2f}, {e[0][1]:.2f}, ..., {e[0][-2]:.2f}, {e[0][-1]:.2f}]', '->', e[1].replace('_', ' ').title())

unkown_person_img = cv2.imread("images/David-Gilmour.jpg")

unkown_person_rep = DeepFace.represent(
    unkown_person_img,
    detector_backend="yunet",
    model_name="Facenet",
    align=True
)

unkown_person_emb = unkown_person_rep[0]["embedding"]

embeddings.append(unkown_person_emb)
names.append("pessoa_desconhecida")

print("Distância euclidiana entre os embeddings:")
for e in vector_db:
    print(f"d(\"Pessoa desconhecida\", \"{e[1].replace('_', ' ').title()}\")","=", f"{verification.find_euclidean_distance(unkown_person_emb, e[0]):.2f}")

embeddings = np.array(embeddings)

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings)

print("Embeddings em 2D")
for e, b in zip(embeddings_2d, names):
    print(b.replace('_', ' ').title(), '->', f'[{e[0]:.2f}, {e[1]:.2f}]')

plt.figure(figsize=(8, 6))
unique_names = list(set(names))
colors = plt.cm.get_cmap('tab10', len(unique_names))
palette = sns.color_palette("hsv", len(unique_names))
color_map = {name: palette[i] for i, name in enumerate(unique_names)}

for i, name in enumerate(names): 
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color_map[name], label=name.replace('_', ' ').title()) 

for i in range(len(embeddings_2d)):
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], names[i].replace('_', ' ').title())

plt.title('Projeção dos embeddings em 2D utilizando t-SNE')
plt.xlabel('Dimensão 1 do t-SNE')
plt.ylabel('Dimensão 2 do t-SNE')

plt.show()
