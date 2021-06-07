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
    return model


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


def save_model(model, fileName):
    model.save(fileName)


def load_model(fileName):
    model = keras.models.load_model(fileName)
    return model


model_file_name = "D:/College/4th year/second term/Computer Vision/Assignments/ass 2/Convolutional-neural-networks/cnn/model.hdf5"
trained_file_name = "D:/College/4th year/second term/Computer Vision/Assignments/ass 2/Convolutional-neural-networks/cnn/trained.hdf5"

# # generate the model and save it in a file    ---- 1
# myModel = generate_model()
# save_model(myModel, model_file_name)
# # myModel.summary()

# load the model to be used      ---- 2.1
# loaded_model = load_model(model_file_name)
# loaded_model.summary()

# # train the model and save it again    ---- 2.2
# (x_train, y_train), (x_test, y_test) = get_data()
# trained_model = train_model(loaded_model, x_train, y_train)
# save_model(trained_model, trained_file_name)

# # load the final model and evaluate the results    ----- 3
# final_model = load_model(trained_file_name)
# final_model.summary()
# print("Test Data: ")
# evaluate_model(final_model, x_test, y_test)
# print("Train Data: ")
# evaluate_model(final_model, x_train, y_train)

