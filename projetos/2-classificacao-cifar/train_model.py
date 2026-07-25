import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# ---------------------------------------------------------------------------
# Projeto 2 — Classificação CIFAR-10
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset CIFAR-10 via tf.keras.datasets.cifar10
#   2. Normalizar as imagens para [0, 1] (shape (32, 32, 3))
#   3. Separar um conjunto de validação
#   4. Incluir data augmentation (ex: layers.RandomFlip, RandomRotation, RandomZoom)
#      aplicada ao conjunto de treino
#   5. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   6. Treinar com EarlyStopping monitorando a perda de validação
#   7. Exibir a acurácia de validação final no terminal
#   8. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

validation_size = int(0.2 * len(x_train))
x_val = x_train[:validation_size]
y_val = y_train[:validation_size]
x_train = x_train[validation_size:]
y_train = y_train[validation_size:]

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
], name="data_augmentation")

inputs = layers.Input(shape=(32, 32, 3))
img = data_augmentation(inputs)

img = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(img) #1
img = layers.BatchNormalization()(img)
img = layers.MaxPooling2D((2, 2))(img)
img = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(img) #2
img = layers.BatchNormalization()(img)
img = layers.MaxPooling2D((2, 2))(img)
img = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(img) #3
img = layers.BatchNormalization()(img)
img = layers.MaxPooling2D((2, 2))(img)
img = layers.Dropout(0.3)(img)

img = layers.Flatten()(img)
img = layers.Dense(128, activation='relu')(img)
img = layers.Dropout(0.4)(img)
outputs = layers.Dense(10, activation='softmax')(img)
model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    x_train, 
    y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stopping]
)

val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

print(f"Acurácia de validação final: {val_acc * 100:.2f}%")

model.save('model.h5')
print(f"Modelo salvo com sucesso em 'model.h5'")
