#This will create/configure/train and test the model then save it in "mnist_model.h5" in the same directory as the code running.
#MLP Multilayer Perceptron neural network - MNIST - A neuron in layer N is connected to all neurons in layer N+1
# Imports
from keras.datasets import mnist,cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout,MaxPooling2D
from utils import get_dataset, generate_attack_tf #This is needed because mnist has 10 classes - classification problem
from keras.models import load_model


def train_and_save(model_name):
    feature_vector_length = 32*32*3
    num_classes = 10

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    X_train = X_train.reshape((X_train.shape[0],32,32,3))
    X_test = X_test.reshape((X_test.shape[0],32,32,3))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)
    
    input_shape = (feature_vector_length,)
    print(f'Feature shape: {input_shape}')
 
    # Create the model
    model = Sequential()

    model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(10 , activation = 'softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=250, verbose=1, validation_split=0.2) # 20% - 80% training and 20%optimization (training samples)


    evaluate(model,X_test,Y_test)
    model.save("./models/"+model_name+"_2.h5")
    print("Saved model to disk")

def evaluate(model,X_test,Y_test):
    test_results = model.evaluate(X_test, Y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
 


if __name__ == '__main__':
    
    (X_train, Y_train), (X_test, Y_test) = get_dataset('cifar10',True,True, False)

    model = load_model('./models/cifar10_1.h5')
    X_test = generate_attack_tf(model,X_test,Y_test,'PGD')
    evaluate(model,X_test,Y_test)
    


