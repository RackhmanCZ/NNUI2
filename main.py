import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential


def plot_training_reusult(epochs, history):
    epochs_range = range(epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def create_dataset(directory, height, width, batch, subset):
    return tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=0.2,
            subset=subset,
            seed=123,
            image_size=(height, width),
            batch_size=batch)


def predict_img(model, class_names, path, height, width):
    img = tf.keras.utils.load_img(
        path, target_size=(height, width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    prediction = model.predict(img_array)

    # normalization of confidence
    prediction = tf.nn.softmax(prediction)

    prediction_class = np.argmax(prediction)
    print(
        "This image most likely belongs to {} with confidence {:.2f}"
        .format(class_names[prediction_class], prediction[0][prediction_class])
    )


if __name__ == '__main__':
    data_dir = "dataset"
    training_dir = "\\training_set"
    validation_dir = "\\validation_set"
    test_valid_path = "cactus.jpg"
    test_no_valid_path = "no_cactus.jpg"

    model_path = "model.tflite"

    batch_size = 32
    img_height = 180
    img_width = 180

    class_names = ["cactus", "no_cactus"]
    epochs = 5

    train_test = True  # True = train & test, False = test only

    if train_test:
        train_ds = create_dataset(data_dir, img_height, img_width, batch_size, "training")

        validation_ds = create_dataset(data_dir, img_height, img_width, batch_size, "validation")

        # optimization
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        # normalization
        normalization_layer = layers.Rescaling(1. / 255)
        normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        normalized_validation_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))

        # augmentation by fliping images
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                  input_shape=(img_height,
                                               img_width,
                                               3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )

        model = Sequential([
            data_augmentation,
            layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(class_names))
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        history = model.fit(
            normalized_train_ds,
            validation_data=normalized_validation_ds,
            epochs=epochs
        )

        plot_training_reusult(epochs, history)

        model.save('model.h5')

    interpreter = tf.keras.models.load_model('model.h5')

    print("Should pass as cactus.")
    predict_img(interpreter, class_names, test_valid_path, img_height, img_width)
    print("Should not pass as cactus.")
    predict_img(interpreter, class_names, test_no_valid_path, img_height, img_width)
