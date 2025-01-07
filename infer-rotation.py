from PIL import Image
import numpy as np

from transformers import ViTImageProcessor, TFViTModel
from keras import layers as L, Model
import argparse

VIT_WEIGHTS_PATH = "weights/model-vit-ang-loss.h5"
BASE_MODEL = "google/vit-base-patch16-224"
IMAGE_SIZE = 224

class Inference:
    def __init__(self):
        self.vit_model = self._load_vit_model()
        self.image_preprocessor = self._load_image_preprocessor()
    
    def predict_rotation(self, image_path):
        X = self._preprocess(image_path)
        y = self.vit_model.predict(X)[0][0]
        return y

    def _preprocess(self, image_path):
        img = Image.open(image_path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img)

        X_vit = self.image_preprocessor.preprocess(images=[img], return_tensors="pt")["pixel_values"]
        return np.array(X_vit)

    def _load_image_preprocessor(self):
        print("Loading Image Preprocessor")
        return ViTImageProcessor.from_pretrained(BASE_MODEL)

    def _load_vit_model(self):
        print("Loading Model")
        vit_base = TFViTModel.from_pretrained(BASE_MODEL)

        img_input = L.Input(shape=(3,IMAGE_SIZE, IMAGE_SIZE))
        x = vit_base.vit(img_input)
        y = L.Dense(1, activation="linear")(x[-1])

        model = Model(inputs=img_input, outputs=y)
        print(model.summary())

        print("Loading Weights")
        model.load_weights(VIT_WEIGHTS_PATH)

        return model

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    args = parser.parse_args()

    model = Inference()
    expected_angle = model.predict_rotation(args.image_path)
    print(f"Predicted angle is about '{expected_angle}' degrees")
