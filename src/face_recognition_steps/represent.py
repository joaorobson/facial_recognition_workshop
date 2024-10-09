from deepface import DeepFace
import cv2


img = cv2.imread("images/David-Gilmour.jpg")

rep = DeepFace.represent(
    img,
    detector_backend="yunet",
    model_name="Facenet",
    align=True
)

print(rep)
