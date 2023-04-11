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
from cleverhans.tf2.attacks.projected_gradient_descent import (projected_gradient_descent,)
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.spsa import spsa
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import pandas as pd
from sklearn import model_selection
from pandas import get_dummies
import ember
import torch

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_attack_tf(model,x,y,attack):

    if(attack not in ['FGSM','CW','PGD',"CKO",'EMBER']):
        raise Exception("Attack not supported")
    if(attack== 'FGSM'):
        x =  fast_gradient_method(model,x,eps=0.1,norm=np.inf,targeted=False)
    if(attack=='CW'):
        x = carlini_wagner_l2(model,x,targeted=False)
    if(attack=='PGD'):
        x = projected_gradient_descent(model, x, 0.05, 0.01, 40, np.inf)
    if(attack =='CKO'):
        adv = []
        for s in x :
            adv.append(reverse_bit_attack(s,500))
        x = np.array(adv)
    if(attack == "EMBER"):
        adv = []
        for s in x :
            adv.append(attack_Ember(s))
        x = np.array(adv)
    print(f'Generated attacks {attack}')
    return x

    

#Returns model according to provided dataset 'this imlementation considers one model per dataset'
def get_model(name):
    if(name not in ['cifar10_1','cuckoo_1','Ember_2','mnist_1','mnist_2','mnist_3']):
         raise ValueError('Model Not Supported')
    return load_model("./models/"+name+".h5")

def attack_Ember(x):
    aux = []
    for i in x :
        if(i>10) :
            i +=  i/100*2
        aux.append(i)
    return np.array(aux)

# Returns dataset and applies transformation according to parameters
def get_dataset(dataset_name, categorical=False):
    

    if(dataset_name == "mnist"):
        (X_train, Y_train), (X_test, Y_test)  = mnist.load_data()
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
    if(dataset_name == "cifar10"):
        (X_train, Y_train), (X_test, Y_test)  = cifar10.load_data()
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255.0
        X_test = X_test/ 255.0
    if(dataset_name =="cuckoo"):
        df = pd.read_csv("./data/cuckoo.csv")
        df_train=df.drop(['Target'],axis=1)
        df_train.fillna(0)
        df['Target'].fillna('Benign',inplace = True)    
        X= df_train.values
        Y=df['Target'].values
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
    if(dataset_name == 'ember'):
        X_train, Y_train, X_test, Y_test = ember.read_vectorized_features("./data/ember2018/")
    
    if categorical :
        if(dataset_name == 'cuckoo') :
            Y_train = pd.get_dummies(Y_train)
            Y_test = pd.get_dummies(Y_test)
        else : 
            Y_train= to_categorical(Y_train)
            Y_test= to_categorical(Y_test)

    return(X_train, Y_train), (X_test, Y_test)

    
  
def reverse_bit_attack(x,Nb):
    '''Generating adversarial samples by randomly flipping 
    Nb bits from 0 to 1
    For cuckoo dataset
    ''' 
    X_test_crafted=x.copy()

    N_flipped=0
    for index in range(x.size):
        if x[index] == 0:
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
  


    
def get_shape(d):
    if(d=='mnist') : 
        return (28,28)
    if(d=='cifar10'):
        return (-1,3,32,32)
    
def dispersation_index(x):
    if(len(x)==0): raise ValueError('Dispersaiton of an empty array')
    if(type(x) == torch.Tensor) : return torch.var(x)* torch.var(x) / torch.mean(x)
    return np.var(x) * np.var(x) /np.mean(x) 


def normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm

def discretize (arr,nb_bins):
    hist, bins = np.histogram(arr, bins=nb_bins)
    discretized = np.digitize(arr, bins)
    return discretized
   
def scotts_rule(data):
    n = len(data)
    if n == 0:
        return 0
    sigma = np.std(data)
    bin_width = 3.5 * sigma / (n ** (1/3))
    num_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_width))
    return num_bins


def plotAcrossPredictions(gt,ben=None,adv=None,Pred_range=0): 
    X = np.arange(Pred_range)
 
    plt.bar(X - 0.2, gt, 0.2, label = 'GroundTruth',color="grey")
    if(ben):
        plt.bar(X , ben, 0.2, label = 'Benign',color="green")
    if(gt) :
        plt.bar(X + 0.2, adv, 0.2, label = 'Adversarial',color ="red")

    plt.xticks(X)
    plt.xlabel("Prediction")
    plt.ylabel("Metric")
    plt.legend()
    plt.show()
