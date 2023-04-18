import numpy as np 
from keras.datasets import mnist
from keras.datasets import cifar10
import os
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
from cleverhans.tf2.attacks.projected_gradient_descent import (projected_gradient_descent,)
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
#from cleverhans.tf2.attacks.spsa import spsa
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import pandas as pd
from sklearn import model_selection
#from pandas import get_dummies
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
    
    if categorical:
        if(dataset_name == 'cuckoo'):
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


def plotAcrossPredictions(gt, metric, ben=None,adv=None,Pred_range=10):#,data='mnist'): 
    X = np.arange(Pred_range)
 
    
    if(adv) :
        if len(adv)==2:
            plt.bar(X - 0.2, adv[0], 0.2, label = 'FGSM',color ="red")
            plt.bar(X , adv[1], 0.2, label = 'PGD',color ="orange")
            
            plt.bar(X+ 0.2 , gt, 0.2, label = 'GroundTruth',color="grey")
            if(ben):
                plt.bar(X+0.4 , ben, 0.2, label = 'Benign',color="green")
        else:
            plt.bar(X - 0.2, adv[0], 0.2, label = 'adv',color ="red")
            plt.bar(X - 0.2, gt, 0.2, label = 'GroundTruth',color="grey")
            if(ben):
                plt.bar(X , ben, 0.2, label = 'Benign',color="green")

    plt.xticks(X)
    plt.xlabel("Prediction")
    plt.ylabel(metric)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
def plotAcrossNodes(gt, metric, ben,adv,Node_range=10):#,data='mnist'): 
    plt.figure(figsize=(30,10))
    X = np.arange(Node_range)
    mask=[]
    
    for i in range(Node_range):
        if gt[i]<0.6 or ben[i]<0.6 or adv[0][i]<0.6 or adv[1][i]<0.6:
            mask.append(False)
        else:
            mask.append(True)
            
    X=X[mask]
    gt=gt[mask]
    adv[0]=adv[0][mask]
    adv[1]=adv[1][mask]
    ben=ben[mask]        
    
    print(len(X))
    
    if len(adv)==2:
        plt.bar(X - 0.1, abs(adv[0] - gt) , 0.2, label = 'FGSM - GT',color ="red")
        plt.bar(X , abs(adv[1] - gt), 0.2 , label = 'PGD - GT',color ="orange")
        
        #plt.bar(X+ 0.2 , gt, 0.2, label = 'GroundTruth',color="grey")
        
        plt.bar(X+0.1 , ben, 0.2, label = 'Benign- GT',color="green")
    else:
        plt.bar(X - 0.1, abs(adv-gt), 0.2, label = 'adv - GT',color ="red")
        #plt.bar(X - 0.2, gt, 0.2, label = 'GroundTruth',color="grey")
        
        plt.bar(X , abs(ben-gt), 0.2, label = 'Benign - GT',color="green")

    plt.xticks(X,rotation = 90,fontsize=5)
    plt.xlabel("Nodes")
    plt.ylabel(metric)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
def plotDiff(FGSM_diff,PGD_diff,adv_diff,Node_range=10):#,data='mnist'): 
    plt.figure(figsize=(30,10))
    X = np.arange(Node_range)
    
    
    
    plt.bar(X - 0.1, FGSM_diff , 0.2, label = 'FGSM - Ben',color ="red")
    plt.bar(X , PGD_diff, 0.2 , label = 'PGD - Ben',color ="orange")
    
    #plt.bar(X+ 0.2 , gt, 0.2, label = 'GroundTruth',color="grey")
    
    plt.bar(X+0.1 , adv_diff, 0.2, label = 'FGSM- PGD',color="blue")
        

    plt.xticks(X,rotation = 90,fontsize=5)
    plt.xlabel("Nodes")
    plt.ylabel('Average Activation Difference')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()