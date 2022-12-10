from Activations import Activations
from Accessor import Accessor
from utils import * 
from numpy import *
from Validator import Validator
from keras.models import load_model
import ember
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout,  Activation
from keras.utils import np_utils
from keras.layers import BatchNormalization
import keras    


import lightgbm as lgb


def model_accuracy() :
    model =get_model("cifar10_1")
    (X_train, Y_train), (X_test, Y_test) = get_dataset("cifar10",True,True,False)

    
    test_results = model.evaluate(X_test, Y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
 
    print('X train and Y train : %s   - %s ' %(X_train.shape,Y_train.shape))
    print('X Test and Y test : %s   - %s ' %(X_test.shape,Y_test.shape))


  
    X_adv = generate_attack_tf(model,X_test,Y_test,attack="PGD")

    X_adv = np.array(X_adv)


    test_results = model.evaluate(X_adv, Y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
    

def train_ember():
    print('a')
    X_train, y_train, X_test, y_test = ember.read_vectorized_features("./data/ember2018/")
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(f'X_train :{X_train.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'X_test  : {X_test.shape}')
    print(f'y_test  :{y_test.shape}')


    batch_size = 500
    epochs = 500
    model = Sequential()

    model.add(Dense(4608, activation='relu', input_shape=(2381,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(3584, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(3072, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(2560, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(1536, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.01))

    model.add(Dense(2, activation='sigmoid'))
    model.summary()
    model.compile(keras.optimizers.Adam(lr=1e-4),loss='binary_crossentropy', metrics=['accuracy'])# cross-entropy
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(X_test, y_test))
    model.save("./models/Ember_1_f.h5")
    
    #validator = Validator(1,1)
    #validator.accuracy_from_file('./adversarial/cuckoo/CKO/cuckoo_1')


if __name__ == '__main__':
 
    X_train, y_train, X_test, y_test = ember.read_vectorized_features("./data/ember2018/")
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(f'X_train :{X_train.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'X_test  : {X_test.shape}')
    print(f'y_test  :{y_test.shape}')
    model = load_model('./models/Ember_2.h5')
    print(X_test[0])
    print(model.predict([X_test[0]]))
