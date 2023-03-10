from Accessor import Accessor
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout,MaxPooling1D,Conv1D
from utils import get_dataset #This is needed because mnist has 10 classes - classification problem
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


    


if __name__ == "__main__":

    adversarial_sample = Accessor('./adversarial/cifar10/FGSM/cifar10_1')
    begning_sample = Accessor('./begnign/cifar10/cifar10_1')
    #alternate_test = Accessor('./adversarial/cifar10/fgsm/cifar10_1')
    expected_nb_nodes = 3978
    begning_sample_act = begning_sample.get_all()
    adv_sample_act = adversarial_sample.get_all()
    print(f'Begn : {len(begning_sample_act)}')
    print(f'Adv : {len(adv_sample_act)}')


    #alternate_act =  alternate_test.get_all()

    X = []
    Y = []

    outlier_counter = 0
    for x,y in zip(begning_sample_act,adv_sample_act):
        outler_counter = 0
        #y.set_layer_range(0,1)  
        #x.set_layer_range(0,1)  
        #x = x.drop_and_get(0.1)
        #y = y.drop_and_get(0.1)

        '''
        if(len(x.flatten()) != 69506  ):
            outlier_counter+=1
            print(len(x.flatten()))
            continue
        '''

        X.append(x.flatten())
        Y.append(0)
        #y.set_layer_range(0,1)
        
        X.append( y.flatten())
        Y.append(1)
        



    
    print('outlier counter %s '%(outlier_counter))
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    X = X.reshape(1,11180)
    print(f'Final sahpe : {X.shape}')
    print(f'Final sahpe : {Y.shape}')
   
    '''
    #Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)
    y_train = to_categorical(y_train)
    print(f'X_train {X_train.shape}')
    print(f'y_train {y_train.shape}')
    print(f'X_test {X_test.shape}')
    print(f'y_test {y_test.shape}')
    '''

    '''
    alternate_x = []
    alternate_y = []

    #ALternate processing :
    for x in alternate_act :
        x.set_layer_range(0,1)
        if(  len(x.flatten()) != expected_nb_nodes  ):
            continue
        alternate_x.append(x.flatten())    
        alternate_y.append(1)
    alternate_x = np.array(alternate_x)
    alternate_y = np.array(alternate_y)
    alternate_y = to_categorical(alternate_y)




    '''
    #Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)
    y_train = to_categorical(y_train)
    print(f'X_train {X_train.shape}')
    print(f'y_train {y_train.shape}')
    print(f'X_test {X_test.shape}')
    print(f'y_test {y_test.shape}')



    model = Sequential()
    model.add(Conv1D(512,kernel_size =1,input_shape=(1,11180),activation='relu'))
    model.add(Conv1D(256,kernel_size =1,activation='relu'))
    model.add(Conv1D(256,kernel_size =1,activation='relu'))
    model.add(Conv1D(128,kernel_size =1,activation='relu'))
    model.add(Conv1D(64,kernel_size =1,activation='relu'))
    model.add(Conv1D(32,kernel_size =1,activation='relu'))
    model.add(Conv1D(12,kernel_size =1,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(2 , activation = 'softmax'))
    model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

    #X_train = np.split(X_train,3)
   
    X_train =np.array(X_train)
    y_test =np.array(y_test)

    model.fit(X_train, y_train, epochs=20)

    #print(' accuracy : %s' %(accuracy_score(pred,y_test)))
       
    

    '''  
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.33)

    clf = tree.DecisionTreeClassifier()
    model = clf.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(' accuracy : %s' %(accuracy_score(pred,y_test)))

    '''
    '''
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.33)
    log_reg_model = LogisticRegression(max_iter=2500,
                                   random_state=42)
    log_reg_model.fit(X_train, y_train)
    pred = log_reg_model.predict(X_test) # Predictions
    print(' accuracy : %s' %(accuracy_score(pred,y_test)))
    '''
    
    '''
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.33)

    model = GaussianNB()

    # Train the model using the training sets
    model.fit(X_train,y_train)

    #Predict Output
    pred =model.predict(X_test) # 0:Overcast, 2:Mild
    
    print(' accuracy : %s' %(accuracy_score(pred,y_test)))
    ''' 

