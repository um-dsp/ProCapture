#This will create/configure/train and test the model then save it in "mnist_model.h5" in the same directory as the code running.
#MLP Multilayer Perceptron neural network - MNIST - A neuron in layer N is connected to all neurons in layer N+1
# Imports
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical #This is needed because mnist has 10 classes - classification problem


def train_and_save(model_name):
    feature_vector_length = 28*28*1
    num_classes = 10 #10

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()


    X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
    X_test = X_test.reshape(X_test.shape[0], feature_vector_length)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)


    input_shape = (feature_vector_length,)
    print(f'Feature shape: {input_shape}')

    # Create the model
    model = Sequential()
    model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=250, verbose=1, validation_split=0.2) # 20% - 80% training and 20%optimization (training samples)
    evaluate(model,X_test,Y_test)
    model.save(model_name+".h5")
    print("Saved model to disk")

def evaluate(model,X_test,Y_test):
    test_results = model.evaluate(X_test, Y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
 


if __name__ == '__main__':
    train_and_save("mnist_2")


