from deepface import DeepFace
import cv2

image = cv2.imread("images/David-Gilmour.jpg")

faces = DeepFace.extract_faces(
    image,
    detector_backend="yunet",
    align=True
)

for face in faces:
    print(face)
    facial_area = face['facial_area']
    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

    start_point = (facial_area['x'], facial_area['y'])
    end_point = (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h'])
    color = (0, 0, 255)  # Red color for the rectangle
    thickness = 10  # Line thickness
    cv2.rectangle(image, start_point, end_point, color, thickness)
    cropped_face = image[y:y+h, x:x+w]

cv2.imwrite('images/detected_face.jpg', image)
cv2.imwrite('images/cropped_face.jpg', cropped_face)
