import cv2
from deepface import DeepFace 
from deepface.modules.detection import detect_faces, align_img_wrt_eyes


img = cv2.imread('images/David-Gilmour.jpg')

face_objs = detect_faces(
    detector_backend="yunet",
    img=img,
    align=False,
    expand_percentage=0,
)

for face_obj in face_objs:
    aligned_img, angle = align_img_wrt_eyes(img=img, left_eye=face_obj.facial_area.left_eye, right_eye=face_obj.facial_area.right_eye)

faces = DeepFace.extract_faces(
    aligned_img,
    detector_backend="yunet",
)

for face in faces:
    print(face)
    facial_area = face['facial_area']
    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

    start_point = (facial_area['x'], facial_area['y'])
    end_point = (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h'])
    color = (0, 0, 255)  # Green color for the rectangle
    thickness = 10  # Line thickness
    cropped_face = aligned_img[y:y+h, x:x+w]

cv2.imwrite('images/cropped_face_aligned.jpg', cropped_face)
