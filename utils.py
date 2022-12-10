from ast import Raise
from pkgutil import get_data
from tokenize import PlainToken
import numpy as np 
import tensorflow as tf

from keras import backend as K

from keras.datasets import mnist
from keras.datasets import cifar10
import os

from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
from Activations import Activations
from cleverhans.tf2.attacks.projected_gradient_descent import (projected_gradient_descent,)
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.spsa import spsa
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import pandas as pd
from sklearn import model_selection
from pandas import get_dummies


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_attack_tf(model,x,y,attack):
    if(attack not in ['FGSM','CW','PGD',"CKO"]):
        raise Exception("Attack not supported")
    if(attack== 'FGSM'):
        x =  fast_gradient_method(model,x,eps=0.1,norm=np.inf,targeted=False)
    if(attack=='CW'):
        x = carlini_wagner_l2(model,x,targeted=False)
    if(attack=='PGD'):
        x = projected_gradient_descent(model, x, 0.05, 0.01, 40, np.inf)
    if(attack =='CKO'):
        x = reverse_bit_attack(x,500)

    return x

    

#Returns model according to provided dataset 'this imlementation considers one model per dataset'
def get_model(name):
        return load_model("./models/"+name+".h5")


# Returns dataset and applies transformation according to parameters
def get_dataset(dataset_name, normalize = True, categorical=False,reshape=False):
    if(dataset_name not in ['mnist','cifar10','cuckoo']):
        raise Exception('Dataset not supported')
    if(dataset_name == "mnist"):
        (X_train, Y_train), (X_test, Y_test)  = mnist.load_data()
        feature_vector_length= 784
    if(dataset_name == "cifar10"):
        (X_train, Y_train), (X_test, Y_test)  = cifar10.load_data()
        feature_vector_length = 32*32*3
    if(dataset_name =="cuckoo"):
        df = pd.read_csv("./data/cuckoo.csv")
        df_train=df.drop(['Target'],axis=1)
        df_train.fillna(0)
        df['Target'].fillna('Benign',inplace = True)    
        X= df_train.values
        Y=df['Target'].values
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
        feature_vector_length =1549




    if normalize : 
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255.0
        X_test = X_test/ 255.0
    if categorical : 
        
        #if type(Y_test[0] == "str"):  
        #    Y_test = get_dummies(Y_test)
        #    Y_train = get_dummies(Y_train)
        #else : 
        Y_train= to_categorical(Y_train,10)
        Y_test= to_categorical(Y_test,10)

    if reshape :
        X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
        X_test = X_test.reshape(X_test.shape[0], feature_vector_length)

    return(X_train, Y_train), (X_test, Y_test)

    
  
def reverse_bit_attack(X_test,Nb):
    '''Generating adversarial samples by randomly flipping 
    Nb bits from 0 to 1
    For cuckoo dataset
    ''' 

    X_test_crafted=X_test.copy()

    N_flipped=0
    for index in range(X_test.size):
        if X_test[index] == 0:
            X_test_crafted[index]=1
            N_flipped+=1
        if N_flipped == Nb:
            break
  

    return X_test_crafted 



        
        
    
    # Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

  

# returns folder name given attack and dataset  
def get_folder_name(attack,dataset):

    if(attack):
        folder = "./adversarial/"+dataset
    else: 
        folder = "./begnign/"+dataset

    # Create the folder if it doeasnt exist
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    if(attack):
        folder += "/"+attack
    
    # Create the folder if it doeasnt exist
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    return folder


# return the activations of each mlayer of the mdel  
def get_layers_activations(model,input):
    inp = model.input                                           
    outputs = [layer.output for layer in model.layers]          
    functors = [K.function([inp], [out]) for out in outputs]  
    layer_outs = [func([input]) for func in functors]
    return layer_outs

def compute_accuracy_tf(model,X_dataset,Y_dataset):
    correct = 0
    for i,x in enumerate(X_dataset):
        x= x.reshape(-1,28,28)
        x= generate_attack_tf(model,x)
        pred = np.argmax(model.predict(x,verbose=0))
        if(pred == Y_dataset[i]):
            correct+=1
    print(correct/Y_dataset.shape[0]*100)
  


#Generates acivations for a given model and input and saves in corresponding folder
def generate_and_save_activations(model,input,index,label,folder_name):
    #input = np.reshape(input,(-1,28,28,1))
    ac =  get_layers_activations(model,input)
    prediction =np.argmax(model.predict(input,verbose=0)[0])
    activations = [item for sublist in ac for item in sublist]
    #print(np.array(activations[0][0]).shape)

    #For string label translated to dummy categories need to format label to string to put in file name
    '''
    if(isinstance(label,pd.Series)):
       label = label['Benign'].astype(str) + label['Malware'].astype(str)
    '''
    list= []
    for i in activations : 
        arr = np.array(i)
        if(len(arr.shape) ==4):
            arr = np.moveaxis(arr, [0,1,2,3], [3,2,1,0])
        if(len(arr.shape)==2):
            arr = np.moveaxis(arr, [0,1], [1,0])
        arr = np.squeeze(arr)
        list.append(arr)
   

    if(len(label) != 1):
        label =np.argmax(label)
    
    #for i in list :
    #    arr = np.asarray(i)
    #    print(arr.shape)

    
    a = Activations(index,label,prediction,list)
    a.save_cnn(list,folder_name)
    return label == prediction 
    
def get_shape(d):
    if(d=='mnist') : 
        return (28,28)
    if(d=='cifar10'):
        return (-1,3,32,32)

'''
def generate_attack(data,model,attack):
    i = 0
    num_correct = 0
    num_samples = 0
    model.eval()
    model =model.to(device)
    x_adv = torch.Tensor().to(device)
    y_adv = torch.Tensor().to(device)
    for x,y in data.test : 
        x = x.to(device=device)
        y = y.to(device=device)
        if(attack == 'FGSM'):
            x = fast_gradient_method(model, x, eps=0.01, norm = np.inf)
        if(attack == "CW"):
            x = carlini_wagner_l2(model, x, 10,y,targeted = False)
        if attack=='SPSA':
            x = spsa(model, x,eps=0.01,nb_iter=500,norm = np.inf,sanity_checks=False)
        if attack=='PGD':
            x = projected_gradient_descent(model, x, 0.01, 0.01, 40, np.inf)
        
        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)
        printProgressBar(i , len(data.test), prefix = 'Progress:', suffix = 'Complete', length = 50)
        i+=1
        print(f' accuracy : {float(num_correct)/float(num_samples)*100:.2f}') 



def compute_accuracy(x,y,model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in zip(x,y):
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f' accuracy : {float(num_correct)/float(num_samples)*100:.2f}') 



'''


def get_input_from_activation(activation):
    set = activation.get_activations_set()

    for i in set :
        print(len(i))
    exit()
    input = set[0] #Layer 0 represents each pixel of the input image
    plt.imshow(np.reshape(input,(28,28)))
    plt.show()


