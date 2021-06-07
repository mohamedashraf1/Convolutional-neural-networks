import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# Model / data parameters
num_classes = 10


def get_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def generate_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def train_model(model, x_train, y_train):
    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


# run starts at 5:36
myModel = generate_model()
# myModel.summary()
(x_train, y_train), (x_test, y_test) = get_data()
train_model(myModel, x_train, y_train)
evaluate_model(myModel,x_test, y_test)
# run ends at 5:44

#
# def generate_model():
#     model = tf.keras.Sequential()
#     # input shape:  H, W, number of channels
#     model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#     model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#
#     model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#
#     model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
#     model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#
#     model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
#     # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
#
#     # model.add(tf.keras.layers.Flatten())
#     #
#     # model.add(tf.keras.layers.Dense(512, activation='relu'))
#     # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#
#     # model.compile()
#     # This builds the model for the first time: model.fit(x, y, batch_size=32, epochs=10) `
#     return model
#
