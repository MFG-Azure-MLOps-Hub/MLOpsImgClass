import os
import glob
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from utils import load_data, one_hot_encode


def train_model(X_train, X_test, y_train, y_test):
    # Can be configured in a seperate file
    n_inputs = 28 * 28
    n_h1 = 300
    n_h2 = 100
    n_outputs = 10
    n_epochs = 5
    batch_size = 20
    learning_rate = 0.001

    y_train = one_hot_encode(y_train, n_outputs)
    y_test = one_hot_encode(y_test, n_outputs)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

    # Build a simple MLP model
    model = Sequential()
    # first hidden layer
    model.add(Dense(n_h1, activation='relu', input_shape=(n_inputs,)))
    # second hidden layer
    model.add(Dense(n_h2, activation='relu'))
    # output layer
    model.add(Dense(n_outputs, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=learning_rate),
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              verbose=2,
              validation_data=(X_test, y_test))
    # callbacks=[LogRunMetrics()])

    score = model.evaluate(X_test, y_test, verbose=0)

    # log a single value
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    metrics = {"acc": score[1]}
    return model, metrics


def main():
    data_folder = "data"
    X_train_path = glob.glob(
        os.path.join(
            data_folder,
            '**/train-images-idx3-ubyte.gz'),
        recursive=True)[0]
    X_test_path = glob.glob(
        os.path.join(
            data_folder,
            '**/t10k-images-idx3-ubyte.gz'),
        recursive=True)[0]
    y_train_path = glob.glob(
        os.path.join(
            data_folder,
            '**/train-labels-idx1-ubyte.gz'),
        recursive=True)[0]
    y_test_path = glob.glob(
        os.path.join(
            data_folder,
            '**/t10k-labels-idx1-ubyte.gz'),
        recursive=True)[0]

    X_train = load_data(X_train_path, False) / 255.0
    X_test = load_data(X_test_path, False) / 255.0
    y_train = load_data(y_train_path, True).reshape(-1)
    y_test = load_data(y_test_path, True).reshape(-1)

    # Train the model
    model, metrics = train_model(X_train, X_test, y_train, y_test)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")

    model_file = "train.h5"
    model.save(model_file)
    model1 = load_model(model_file)
    print(model1.summary())


if __name__ == '__main__':
    main()
