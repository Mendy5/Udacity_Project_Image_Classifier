import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import argparse
import numpy as np
import json
from pre_processing_image import process_image



parser = argparse.ArgumentParser(description='Read Arguments from Terminal')
parser.add_argument('image_path', help='image path', default='')
parser.add_argument('model_path', help='model path', default='')
parser.add_argument('--top_k', help='top K value', default='5')
parser.add_argument('--category_names', help='get class label', default = None)
args = parser.parse_args()

IMAGE_PATH = args.image_path
MODEL_PATH = args.model_path
TOP_K = args.top_k
CATEGORY_NAMES = args.category_names

def load_model(model_path):
    model = keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
    return model

def predict(image_path, model, top_k):
    #get and process images
    image = process_image(image_path)

    #predict image
    top_k = int(top_k)
    prediction = model.predict(image)
    probs, classes = tf.math.top_k(prediction, k=top_k)
    probs = probs.numpy()[0]
    classes = [str(num + 1) for num in classes.numpy()[0]]
    return probs, classes


def main(image_path, model_path, top_k, category_names):

    my_model = load_model(model_path)
    probs, classes = predict(image_path, my_model, top_k)

    if category_names != None:
        label_map = category_names
        with open(label_map, 'r') as f:
             class_names = json.load(f)
        classes = [class_names[label] for label in classes]

    return probs, classes

if __name__ == "__main__":
    probs, classes = main(IMAGE_PATH, MODEL_PATH, TOP_K, CATEGORY_NAMES)
    print(probs)
    print(classes)
