from datetime import datetime
import random
from io import BytesIO
from typing import Optional, List

import tensorflow as tf
from fastapi import FastAPI
import base64

from api import settings

app = FastAPI()

# model = tf.keras.models.load_model('/Users/arnasmitkevicius/PycharmProjects/image-classification/saved_model/LTVLN2/v1')


from pydantic import BaseModel, BaseSettings

from PIL import ImageFile, Image

# ImageFile.LOAD_TRUNCATED_IMAGES = True



items = {}


@app.on_event("startup")
async def startup_event():
    for i in settings.settings:
        items[i["area"]] = tf.keras.models.load_model(i["path"])


class Item(BaseModel):
    image: Optional[str] = None
    area: Optional[str] = None


def recognize(res, url):
    model = items[res["area"]]

    img = Image.open(BytesIO(base64.b64decode(url)))
    img = img.resize((512, 512))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    score = score.numpy().tolist()

    print(score)
    print(res["labels"])

    result = {}

    for index, i in enumerate(score):
        result[res["labels"][index]] = i

    return result


@app.post("/")
def recognition(image: Item):
    res = next(filter(lambda x: x["area"] == image.area, settings.settings), None)

    return recognize(res, image.image)


@app.get("/test")
def recognition(image: Item):
    return image
