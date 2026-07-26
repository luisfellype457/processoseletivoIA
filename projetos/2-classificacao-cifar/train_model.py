import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

try:
    import tf_keras as keras
    from tf_keras import layers, models, callbacks
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

validation_size = int(0.2 * len(x_train))
x_val = x_train[:validation_size]
y_val = y_train[:validation_size]
x_train = x_train[validation_size:]
y_train = y_train[validation_size:]

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
], name="data_augmentation")

x_train_augmented = data_augmentation(x_train)

inputs = layers.Input(shape=(32, 32, 3))

x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    x_train_augmented,
    y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stopping]
)

val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"Acurácia de validação final: {val_acc * 100:.2f}%")

keras.models.save_model(model, 'model.h5', save_format='h5')
print("Modelo salvo com sucesso em 'model.h5'")
