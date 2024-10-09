# Oficina de reconhecimento facial

## Configuração inicial

* Clone o repositório:

```
git clone https://github.com/joaorobson/facial_recognition_workshop
```

* Instale as dependências:

```
cd facial_recognition_workshop
pip install -r requirements.txt

```

## Etapas do reconhecimento facial

O processo de reconhecimento facial consiste em 5 etapas: detecção, alinhamento, normalização, representação e verificação.
Cada etapa está implementada em um _script_ independente em `src/face_recognition_steps`.

Para executá-los, siga o exemplo abaixo:

```
python src/face_recognition_steps/detect.py
```


## API

Para executar a API localmente, execute:

```
fastapi dev src/api/main.py
```

## Front-end

Para executar a interface gráfica com streamlit,execute:

```
streamlit run src/frontend/main.py
```
