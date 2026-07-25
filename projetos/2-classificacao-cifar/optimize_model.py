import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras as keras

def main():
    model_path = 'model.h5'
    print(f"Carregando {model_path}...")
    model = keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
        
    print("Sucesso! Arquivo model.tflite gerado.")

if __name__ == "__main__":
    main()
