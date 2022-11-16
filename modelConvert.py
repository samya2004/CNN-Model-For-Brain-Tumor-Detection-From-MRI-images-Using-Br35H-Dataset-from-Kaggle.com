import keras
from keras.models import load_model

model=load_model('brainTumor20epochsBinary.h5')
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_lite_model=converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tf_lite_model)