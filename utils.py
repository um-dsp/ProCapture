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

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_attack_tf(model,x,y,attack):
    if(attack== 'FGSM'):
        x =  fast_gradient_method(model,x,eps=0.05,norm=np.inf,targeted=False)
    if(attack=='CW'):
        x = carlini_wagner_l2(model,x,targeted=False)
    if(attack=='PGD'):
        x = projected_gradient_descent(model, x, 0.05, 0.01, 40, np.inf)
    return x
    

#Returns model according to provided dataset 'this imlementation considers one model per dataset'
def get_model(dataset):
    if(dataset == 'mnist'):
        return load_model("./models/mnist.h5")
    if(dataset =='cifar10'):
        return load_model("./models/cifar10vgg.h5")


# Returns dataset and applies transformation according to parameters
def get_dataset(dataset_name, normalize = True, categorical=False,reshape=False):
    if(dataset_name == "mnist"):
        (X_train, Y_train), (X_test, Y_test)  = mnist.load_data()
        feature_vector_length= 784
    if(dataset_name == "cifar10"):
        (X_train, Y_train), (X_test, Y_test)  = cifar10.load_data()
        feature_vector_length = 32*32*3
    if normalize : 
        X_train = X_train / 255.0
        X_test = X_test/ 255.0
    if categorical : 
        Y_test= to_categorical(Y_test)
    if reshape :
        X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
        X_test = X_test.reshape(X_test.shape[0], feature_vector_length)

    return(X_train, Y_train), (X_test, Y_test)

    
  



        
        
    
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
    input = np.reshape(input,(-1,28,28))

    ac =  get_layers_activations(model,input)
    prediction =np.argmax(model.predict(input,verbose=0)[0])
    activations = [item for sublist in ac for item in sublist]
    a = Activations(index,label,prediction,activations)
    a.save_csv(folder_name)
    return a

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



