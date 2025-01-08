from PIL import Image

import numpy as np
from numpy.typing import NDArray

from transformers import TFViTModel
import keras
import argparse

VIT_WEIGHTS_PATH = "weights/model-vit-ang-loss.h5"
BASE_MODEL = "google/vit-base-patch16-224"
IMAGE_SIZE = 224

def to_numpy(img: Image.Image) -> NDArray[np.float32]:
    return np.asarray(img if img.mode == "RGB" else img.convert("RGB"), dtype=np.float32) / 255.0

def normalize(
    img: NDArray[np.float32], mean: float | NDArray[np.float32], std: float | NDArray[np.float32]
) -> NDArray[np.float32]:
    return np.divide(img - mean, std, dtype=np.float32)

class Inference:
    def __init__(self):
        self.vit_model = self._load_vit_model()
    
    def predict_rotation(self, image_path):
        X = self._preprocess(image_path)
        y = self.vit_model.predict(X)[0][0]
        return y

    def _preprocess(self, image_path):
        image = Image.open(image_path)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_np = to_numpy(image)
        image_np = normalize(image_np, 0.5, 0.5)
        return np.expand_dims(image_np.transpose(2, 0, 1), 0)

    def _load_vit_model(self):
        print("Loading Model")
        vit_base = TFViTModel.from_pretrained(BASE_MODEL)

        img_input = keras.layers.Input(name="image", shape=(3,IMAGE_SIZE, IMAGE_SIZE))
        x = vit_base.vit(img_input)
        y = keras.layers.Dense(1, activation="linear")(x[-1])

        model = keras.Model(inputs=img_input, outputs=y)
        model.load_weights(VIT_WEIGHTS_PATH)
        print(model.summary())

        # model.save("saved_model")

        return model

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    args = parser.parse_args()

    model = Inference()
    expected_angle = model.predict_rotation(args.image_path)
    print(f"Predicted angle is about '{expected_angle}' degrees")
