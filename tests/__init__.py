import unittest
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from parameterized import parameterized

class_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "NO_OBJECT"]

# model = tf.keras.models.load_model('/Users/arnasmitkevicius/PycharmProjects/image-classification/saved_model/v1')

images = [
    "https://lnm.lt/wp-content/uploads/2021/04/LNM-web-ikonos-3.jpg",
    "https://pilotas.lt/wp-content/uploads/2018/12/gedim_vn_150600_e02_BRA.jpg",
    "https://dirbtuves.com/image/cache/drobes/Kvadratines/DRK000004s-600x600.jpg"
    "http://static.flickr.com/110/296447620_0c62f71758.jpg",
    "https://www.govilnius.lt/images/5e10958b5225306761d7a916?w=750&h=500",
    "https://madeinvilnius.lt/wp-content/uploads/2016/12/bigstock-145318535.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/45/Vilnius_%28Wilno%29_-_cathedral.jpg",
    "https://www.vle.lt/tmp/vle-images/107802_3.jpg",
    "https://s1.15min.lt/images/photos/2014/02/04/original/vilniaus-arkikatedra-2014-uju-vasari-52f0cbf2a4073.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/a/aa/Palace_of_the_Grand_Dukes_of_Lithuania_2019_2.jpg",
    "https://www.govilnius.lt/images/5e35ee41f646c6bc7f5d70bf?w=750&h=500",
    "https://static.etaplius.lt/media/etaplius_gallery_image/5c76c3725008a/nuotrauka.jpg",
    "https://www.govilnius.lt/images/5df3c8608ef659833128921d?w=750&h=500",
    "https://img.geocaching.com/8d7503d3-3531-4390-92ac-5ed7392ba17e.jpg"
]


# for image in images:
#     try:
#         sunflower_path = tf.keras.utils.get_file(
#             datetime.now().strftime("%Y%m%d-%H%M%S") + str(random.randint(0,1000)),
#             origin=image)
#
#         img = tf.keras.utils.load_img(
#             sunflower_path, target_size=(256, 256)
#         )
#         img_array = tf.keras.utils.img_to_array(img)
#         img_array = tf.expand_dims(img_array, 0)  # Create a batch
#
#         predictions = model.predict(img_array)
#         score = tf.nn.softmax(predictions[0])
#
#         print(
#             "This image most likely belongs to {} with a {:.2f} percent confidence."
#                 .format(class_names[np.argmax(score)], 100 * np.max(score))
#         )
#
#         print(score)
#     except Exception as e:
#         print(e)


def detectImage(model, image):
    sunflower_path = tf.keras.utils.get_file(
        datetime.now().strftime("%Y%m%d-%H%M%S") + str(random.randint(0, 1000)),
        origin=image)

    img = tf.keras.utils.load_img(
        sunflower_path, target_size=(512, 512)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return class_names[np.argmax(score)]


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = tf.keras.models.load_model(
            '/Users/arnasmitkevicius/PycharmProjects/image-classification/main/saved_model/LTVLN2/v1')


class TestInit(TestModel):
    @parameterized.expand([
        ["detects g. tower", "https://lnm.lt/wp-content/uploads/2021/04/LNM-web-ikonos-3.jpg", "1"],
        ["detects g. tower", "https://pilotas.lt/wp-content/uploads/2018/12/gedim_vn_150600_e02_BRA.jpg", "1"],
        ["detects g. tower", "https://dirbtuves.com/image/cache/drobes/Kvadratines/DRK000004s-600x600.jpg", "1"],
        ["detects cathedral", "https://upload.wikimedia.org/wikipedia/commons/4/45/Vilnius_%28Wilno%29_-_cathedral.jpg", "3"],
        ["detects cathedral", "https://www.vle.lt/tmp/vle-images/107802_3.jpg", "3"],
        ["detects cathedral", "https://s1.15min.lt/images/photos/2014/02/04/original/vilniaus-arkikatedra-2014-uju-vasari-52f0cbf2a4073.jpg", "3"],
        ["detects grand duke", "https://upload.wikimedia.org/wikipedia/commons/a/aa/Palace_of_the_Grand_Dukes_of_Lithuania_2019_2.jpg", "2"],
        ["detects grand duke", "https://www.govilnius.lt/images/5e35ee41f646c6bc7f5d70bf?w=750&h=500", "2"],
        ["detects grand duke", "https://static.etaplius.lt/media/etaplius_gallery_image/5c76c3725008a/nuotrauka.jpg", "3"],
        ["detects other", "https://www.govilnius.lt/images/5df3c8608ef659833128921d?w=750&h=500", "4"],
        ["detects other", "https://img.geocaching.com/8d7503d3-3531-4390-92ac-5ed7392ba17e.jpg", "4"],
        ["detects other", "https://www.lrt.lt/img/2020/05/19/656221-508556-756x425.jpg", "4"]
    ])
    def test_object_recognition(self, name, input, expected):
        result = detectImage(self.model, input)
        self.assertEqual(expected, result)
