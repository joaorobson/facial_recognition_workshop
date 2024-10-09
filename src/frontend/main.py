import streamlit as st
import base64
import requests
from deepface import DeepFace
from deepface.modules.verification import find_distance
import numpy as np
import cv2

from PIL import Image

def encode_image_to_base64(image):
    return base64.b64encode(image.read()).decode('utf-8')

def send_image_to_endpoint(encoded_image, endpoint_url):
    headers = {'Content-Type': 'application/json'}
    payload = {'base64': encoded_image}
    response = requests.post(endpoint_url, json=payload, headers=headers)       
    st.session_state.response = response


st.set_page_config(layout = "wide")

def facial_recognition():

    st.title("Reconhecimento facial")

    col1, col2, col3 = st.columns(3)
    uploaded_file = None
    button_clicked = False
    encoded_image = b''
    endpoint_url = ''
    image_sent = False

    with col1:
        uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Imagem selecionada", width=200)

            encoded_image = encode_image_to_base64(uploaded_file)
            
            st.text_area("Imagem em base64", encoded_image)

            endpoint_url = st.text_input("Endpoint", "http://localhost:8000/recognize-face")

    with col2:
        if uploaded_file:
            image_sent = st.button("Enviar imagem",
                            use_container_width=True,
                            on_click=send_image_to_endpoint,
                            args=(encoded_image, endpoint_url))

    with col3:
        if uploaded_file and image_sent:
            response = st.session_state.get("response")
            if response.status_code == 200:
                st.success("Image enviada com sucesso!")
                response_json = response.json()
                person_name = response_json.get("name")
                distance = response_json.get("distance")
                st.write("**Pessoa identificada:**", person_name)
                st.write("**Distância para face mais próxima:**", distance)
            else:
                st.error(f"Falha no envio da imagem. Status: {response.status_code}")

def face_detection():

    def draw_bounding_boxes(image, detections):
        for det in detections:
            cv2.rectangle(image, (det["facial_area"]["x"], det["facial_area"]["y"]), 
                                 (det["facial_area"]["x"] + det["facial_area"]["w"], 
                                 det["facial_area"]["y"] + det["facial_area"]["h"]), (255, 0, 0), 10)

        return image 

    st.title("Detecção facial")

    col1, col2, col3 = st.columns(3)
    image1 = None
    image2 = None
    button_clicked = False
    encoded_image = b''
    endpoint_url = ''
    image_sent = False

    with col1:
        uploaded_image = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"], key="image")

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Imagem selecionada", width=200)

            encoded_image = encode_image_to_base64(uploaded_image)

            st.text_area("Imagem em base64", encoded_image)

            endpoint_url = st.text_input("Endpoint", "http://localhost:8000/detect-face")

            image_sent = False

    with col2:
        if uploaded_image:
            image_sent = st.button("Enviar imagem",
                            use_container_width=True,
                            on_click=send_image_to_endpoint,
                            args=(encoded_image, endpoint_url))

    with col3:
        if uploaded_image and image_sent:
            response = st.session_state.get("response")
            if response.status_code == 200:
                st.success("Image enviada com sucesso!")
                response_json = response.json()
                image_arr = np.array(Image.open(uploaded_image).convert('RGB'))
                image_with_boxes = draw_bounding_boxes(image_arr, response_json)
                st.session_state.image_with_boxes = image_with_boxes
                if response_json:
                    st.image(image_with_boxes, caption='Processed Image', use_column_width=True)
                    st.success("Face encontrada!")
                else:
                    st.error("Face não encontrada!")
            else:
                st.error(f"Falha no envio da imagem. Status: {response.status_code}")

page_names_to_funcs = {
    "Detecção facial": face_detection,
    "Reconhecimento facial": facial_recognition,
}

task = st.sidebar.selectbox("Escolha uma tarefa", page_names_to_funcs.keys())
page_names_to_funcs[task]()    
