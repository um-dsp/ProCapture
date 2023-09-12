#This will create/configure/train and test the model then save it in "mnist_model.h5" in the same directory as the code running.
#MLP Multilayer Perceptron neural network - MNIST - A neuron in layer N is connected to all neurons in layer N+1
# Imports
from keras.datasets import mnist,cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout,MaxPooling2D
from library.utils import get_dataset, generate_attack_tf #This is needed because mnist has 10 classes - classification problem
from keras.models import load_model
from keras.layers import BatchNormalization
import torch.nn as nn
import torch.nn.functional as F
import ember
import keras
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import torch
import torch.optim as optim
import tqdm
import copy




def train_Cifar10(model_name):
    
    '''Cifar10 classification model using keras'''
    
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
    print("Trained and Saved CIFAR10 Model")
    
def evaluate(model,X_test,Y_test):
    '''evaluate a keras model'''
    try:
        test_results = model.evaluate(X_test, Y_test, verbose=1)
    except:
        Y_test=tf.math.argmax(Y_test,axis=1)
        test_results = model.evaluate(X_test, Y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
 
   

def train_ember():
    
    '''train Ember classification model'''
    
    X_train, y_train, X_test, y_test = ember.read_vectorized_features("./data/ember2018/")
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(f'X_train :{X_train.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'X_test  : {X_test.shape}')
    print(f'y_test  :{y_test.shape}')


    batch_size = 500
    epochs = 5
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
    model.save("./models/Ember_1_2.h5")
    
    #validator = Validator(1,1)
    #validator.accuracy_from_file('./adversarial/cuckoo/CKO/cuckoo_1')


class Deep(nn.Module):
    def __init__(self,nb_nodes):
        super().__init__()
        self.layer1 = nn.Linear(nb_nodes, 256)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Sequential( 
            nn.Dropout(), 
            nn.Linear(256, 128))
        self.act2 = nn.ReLU()
        self.layer3 = nn.Sequential( 
            nn.Dropout(),
            nn.Linear(128, 64))
        self.act3 = nn.ReLU()
        self.output = nn.Sequential( 
            nn.Dropout(),
            nn.Linear(64, 1))
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
    


class BinaryClassification(nn.Module):
    def __init__(self,nb_nodes):
        super(BinaryClassification, self).__init__()
        
        self.layer_1 = nn.Linear(nb_nodes, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.layer_out(x))
        
        return x
    
def compute_mismatch(model,X_test,Y_test):
    ones= 0
    zeros = 0
    correct = 0 
    pred = model.predict(X_test, verbose=0)
    for x,y in zip(pred,Y_test):
        if(np.argmax(x) == np.argmax(y)):
            correct+=1
        if(np.argmax(x) != np.argmax(y)):
            if(np.argmax(x)== 1):
                ones+=1
            if(np.argmax(x)== 0):
                zeros+=1
    print(f' mismatches 0 {zeros}' )        
    print(f' mismatches 1 {ones}' )        
    print(f' Accuracy  {correct/len(X_test) * 100}' )    

'''
def train_Detection_Model (X,Y):
    Y = to_categorical(Y)
    model = NeuralNetCifar_10()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    accuracy = 0
    #while(accuracy <80):
    X_train, X_test,y_train, y_test = train_test_split(X,Y ,random_state=104, test_size=0.25, shuffle=True)
    print(f'X_test len {len(X_test)}')
    print(f'Y_test len {len(y_test)}')

    accuracy = 1
        #Train Model
    while(accuracy < 95):
        #Load in the data in batches using the train_loader object
        correct =0
        for x, y in  zip(X_train,y_train):  

            y= torch.tensor(y)
            x = x[None, :]
            # Forward pass
            outputs = model(x)
            outputs = torch.squeeze(outputs)

            #print(torch.argmax(outputs),torch.argmax(y))

            loss = criterion(outputs, y)
            correct += 1 if (torch.argmax(outputs) == torch.argmax(y)) else 0 
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'accuracy {correct/len(X_train) *100}')
        accuracy = correct/len(X_train) *100

    torch.save(model, './advDetectionModels/torch_Cifar10_2.pt')
'''

def binary_acc(y_pred, y_test):
    #y_pred_tag = torch.round(torch.sigmoid(y_pred))
    y_pred_tag = torch.round(y_pred)
    
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def train_on_activations(X_train,Y_train,X_test,Y_test,model_name,model_path):
    
    if os.path.exists(model_path):
        print('loading existing model ...')
        model =torch.load(model_path)
    else:
        model=BinaryClassification(X_train.shape[1])
        '''
        if model_name=='cifar10_1':
            model = NeuralNetCifar_10()
        elif model_name=='Ember_2 ':
            model = NeuralNetCifar_10()
        elif model_name=='mnist_1':
            #model=NeuralNetMnist_1()
            model=Deep(X.shape[1])
        elif model_name=='mnist_2':
            model = NeuralNetMnist_2
        elif model_name=='mnist_3':
            model = NeuralNetMnist_3
        elif model_name=='cuckoo_1':
            model = NeuralNetCuckoo_1
        '''

    #Y = to_categorical(Y)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss() 
    #criterion = nn.BCEWithLogitsLoss()

    # shuffling training data
    X_train, _, y_train, _ = train_test_split(X_train,Y_train ,random_state=104, test_size=1, shuffle=True)
    # Standersize data
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    
    if torch.cuda.is_available():
        model=model.cuda()
        
    X_test=torch.Tensor(X_test)
    Y_test=torch.Tensor(Y_test)  
    #print('[Graph MODEL TRAINING]')
    #print(f'X_train len {X_train.shape}')
    #print(f'Y_train len {y_train.shape}')
    #print(f'X_test len {X_test.shape}')
    #print(f'Y_test len {Y_test.shape}')

    n_epoch = 30 # number of epochs to run
    batch_size = 100#00  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    epoch=1
    
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    while(epoch < n_epoch):
        model.train()
        
        with tqdm.tqdm(batch_start, unit="batch") as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                
                
                # take a batch
                x = X_train[start:start+batch_size]
                y = y_train[start:start+batch_size]
                if x.shape[0]==1:
                    continue
                x=torch.Tensor(x)
                y=torch.Tensor(y)
                
                if torch.cuda.is_available():
                    x=x.cuda()
                    y=y.cuda()
                
                optimizer.zero_grad()
                
                #print(type(x))
                y_pred = model(x)
                
                loss = criterion(y_pred, y.unsqueeze(1))
                #acc = binary_acc(y_pred, y.unsqueeze(1))
                
                loss.backward()
                optimizer.step()
                
                '''
                # Forward pass
                outputs = model(x)[:,0]#.detach()[0]
                loss = criterion(outputs, y)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print progress
                acc = (outputs.round() == y).float().mean()
                '''
                bar.set_postfix(
                    loss=float(loss),
                    #train_acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        acc = binary_acc(y_pred, Y_test.unsqueeze(1))#(y_pred.round() == Y_test).float().mean()
        acc = float(acc)
        print('Current test accuracy: ',float(acc))
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        epoch+=1
        
    # restore and save model
    model.load_state_dict(best_weights)
    torch.save(model, model_path)   
    print(f' Best Accuracy : {best_acc}% Saved to {model_path}')

    return model
