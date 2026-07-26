import tensorflow as tf
from tensorflow import keras

def main():
    model_path = 'model.h5'
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
        
    print("Arquivo model.tflite gerado.")

if __name__ == "__main__":
    main()
