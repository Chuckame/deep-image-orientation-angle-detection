import os
from PIL import Image
import numpy as np

from transformers import ViTImageProcessor, TFAutoModel
from tensorflow.keras.models import Model
from tensorflow.keras import layers as L
import argparse

VIT_WEIGHTS_PATH = "weights/model-vit-ang-loss.h5"
IMAGE_SIZE = 224

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

def preprocess(image_path):
    img = Image.open(image_path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img)

    X_vit = feature_extractor(images=[img], return_tensors="pt")["pixel_values"]
    return np.array(X_vit)

def load_vit_model():
    print("Loading Model")
    vit_base = TFAutoModel.from_pretrained("google/vit-base-patch16-224")

    img_input = L.Input(shape=(3,IMAGE_SIZE, IMAGE_SIZE))
    x = vit_base(img_input)
    y = L.Dense(1, activation="linear")(x[-1])

    model = Model(img_input, y)
    print(model.summary())

    print("Loading Weights")
    model.load_weights(VIT_WEIGHTS_PATH)

    return model

class Inference:
    def __init__(self):
        self.vit_model = load_vit_model()
    
    def predict_rotation(self, image_path):
        X = preprocess(image_path)
        y = self.vit_model.predict(X)[0][0]
        return y

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    args = parser.parse_args()

    model = Inference()
    expected_angle = model.predict_rotation(args.image_path)
    print(f"Predicted angle is about '{expected_angle}' degrees")
