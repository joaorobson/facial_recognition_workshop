from pydantic import BaseModel

class Image(BaseModel):
    base64: str

class Person(BaseModel):
    name: str
    distance: float
