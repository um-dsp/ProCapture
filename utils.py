from pkgutil import get_data
from tokenize import PlainToken
import numpy as np 
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from keras import backend as K

from keras.datasets import mnist
from keras.datasets import cifar10
import torch

from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt


def generate_fgsm(image,label,model,epsilon=0.05):

    logits_model = tf.keras.Model(model.input,model.layers[-1].output)

    image = tf.convert_to_tensor(image.reshape((1,28,28))) #The .reshape just gives it the proper form to input into the model, a batch of 1 a.k.a a tensor
    label = np.reshape(label, (1,)).astype('int64') # Give label proper shape and type for cleverhans

    adv_example_untargeted_label = fast_gradient_method(logits_model, image, epsilon, np.inf, targeted=False)
  
    return adv_example_untargeted_label


def generate_carlini_wagner(image,label,model):
    image = torch.tensor(image)
    label = torch.tensor(np.reshape(label, (1,)).astype('int64')) # Give label proper shape and type for cleverhans
    logits_model = tf.keras.Model(model.input,model.layers[-1].output)
    image = carlini_wagner_l2(logits_model, image, 10,label,targeted = False)

    return image


def get_model(dataset):
    predfined_model = ['mnist']
    if( dataset not in predfined_model):
          raise Exception("No predefined model for selected dataset")
    
    return load_model("./models/"+dataset+".h5")

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
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

  
def get_folder_name(attack):
    if(attack):
        folder = "./adversarial/"+attack
    else: 
        folder = "./begnign"
    return folder


   
def get_layers_activations(model,input):
    input = np.reshape(input,(-1,28,28))
    inp = model.input                                           
    outputs = [layer.output for layer in model.layers]          
    functors = [K.function([inp], [out]) for out in outputs]    
    layer_outs = [func([input]) for func in functors]
    return layer_outs











