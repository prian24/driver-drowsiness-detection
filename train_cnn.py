"""Train a small CNN using folders: dataset/train/{closed,open} and dataset/val/{closed,open}."""
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

DATA = Path("dataset")
IMG_SIZE = (64, 64)
BATCH = 32
EPOCHS = 20

def main():
    train = tf.keras.utils.image_dataset_from_directory(
        DATA / "train", labels="inferred", label_mode="binary",
        class_names=["open", "closed"], color_mode="grayscale",
        image_size=IMG_SIZE, batch_size=BATCH, validation_split=None, shuffle=True)
    val = tf.keras.utils.image_dataset_from_directory(
        DATA / "val", labels="inferred", label_mode="binary",
        class_names=["open", "closed"], color_mode="grayscale",
        image_size=IMG_SIZE, batch_size=BATCH, shuffle=False)
    train = train.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)
    val = val.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y)).prefetch(tf.data.AUTOTUNE)

    model = models.Sequential([
        layers.Input((64,64,1)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dropout(0.4), layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=EPOCHS,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    Path("models").mkdir(exist_ok=True)
    model.save("models/eye_state_cnn.keras")
    print("Saved models/eye_state_cnn.keras")

if __name__ == "__main__":
    main()
