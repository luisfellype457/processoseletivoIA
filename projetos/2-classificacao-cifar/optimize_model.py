import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

def main():
    model_path = 'model.h5'
    model = keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
        
    print("Arquivo model.tflite gerado.")

if __name__ == "__main__":
    main()
