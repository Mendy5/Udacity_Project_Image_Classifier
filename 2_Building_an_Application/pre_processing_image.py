import tensorflow as tf
from PIL import Image
import numpy as np

def process_image(image_path):
    #open image from path dir and convert it to numpy array
    im = Image.open(image_path)
    image = np.asarray(im)
    
    #pre-procesing image and normalize it
    image_size = 224
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    
    
    #add dimension
    image = np.expand_dims(image, axis = 0)
    return image

