# Oficina de reconhecimento facial

## Instalação do Python no Windows

Baixe a [versão 3.9](
https://www.python.org/ftp/python/3.9.12/python-3.9.12-amd64.exe)


## Configuração inicial

* Clone o repositório:

```
git clone https://github.com/joaorobson/facial_recognition_workshop
```

* Instale as dependências:

```
cd facial_recognition_workshop
py -3.9 -m venv env
env\Scripts\activate.bat
py -3.9 -m pip install -r requirements.txt
```

## Etapas do reconhecimento facial

O processo de reconhecimento facial consiste em 5 etapas: detecção, alinhamento, normalização, representação e verificação.
Cada etapa está implementada em um _script_ independente em `src/face_recognition_steps`.

Para executá-los, siga o exemplo abaixo:

```
py -3.9 src\face_recognition_steps\detect.py
```


## API

**Antes de executar a API, crie o banco de dados com os _embeddings_**:
```
py -3.9 src\face_recognition_steps\create_vector_db.py
```


Para executar a API localmente, execute:

```
fastapi dev src/api/main.py
```

## Testar API

Para testar a API, é possível utilizar alguma ferramenta específica (ex.: PostMan) ou executar um código Python de testes.

No diretório `src/test_api`, há alguns _scripts_ para testar a API executada localmente.

Para chamar os _endpoints_ básicos, execute:

```
py -3.9 src/test_api/call_basic_endpoints.py
```

Para testar o _endpoint_ de detecção facial, execute:

```
py -3.9 src/test_api/detect_face_using_api.py
```

## Front-end

Para executar a interface gráfica com streamlit,execute:

```
streamlit run src/frontend/main.py
```
