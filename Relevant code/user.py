from generate_activations import generate_activations
from utils import get_dataset,get_model,generate_attack_tf
import numpy as np


'''
get_dataset() in utils would return a dataset of your choice, 
Datasets implemented are mnist , cifar , ember and cuckoo,
the function will returna partitioned (X_train, y_train), (X_test, y_test) set for training and testing set
Other params of this function are  Normalization and to_categorical which will be executed if their flag is set to True



get_model() wil lretreive a model of your choice, the models are in ./models folder
pre-implemented models are mnist_1 , mnist_2,mnist_3,Ember_2 , cifar10 and cuckoo,
get model takes as input a string containing the name of the model

generate_attack_tf() : contains all attacks supported by this software
FGSM, PGD fro cifra10 and mnist
CKO for cuckoo
and EMBER for ember dataset

input :( model:tf.model , X:features , Y: labels , s : attack_name)


generate_activations(): 
will compute and save the activations of a dataset, it takes as input : 
(X : features, Y:labels, model: tf.model, s : file _path to save activations in )


utils.py containes some helpers function that could help you with other functionalities such as 
evaluating the models
'''



if __name__ == "__main__":

    #Use the abose function to generate activations for different datasets/attacks
    exit()    

# This is some sample code to look at if needed
def sampleCode():

    # Mnist generation Code
    (X_train, y_train), (X_test, y_test) = get_dataset('mnist',False,True)
    model = get_model('mnist_1')
    X_adv = generate_attack_tf(model,X_test,y_test,'FGSM')
    generate_activations(X_adv,y_test,model,'./adversarial/test')

    # Cifar generation Code

    (X_train, y_train), (X_test, y_test) = get_dataset('cifar10',True,True)
    model = get_model('cifar10_1')
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test/ 255.0
    X_adv = generate_attack_tf(model,X_test,y_test,'FGSM')
    generate_activations(X_adv,y_test,model,'./adversarial/test')

    # Cuckoo generation Code

    (X_train, y_train), (X_test, y_test) = get_dataset('cuckoo',True,True)
    model = get_model('cuckoo_1')
    print(X_test.shape)
    X_adv = generate_attack_tf(model,X_test,y_test,'CKO')
    generate_activations(X_test,y_test,model,'./adversarial/test')

    #Ember generation Code
    (X_train, y_train), (X_test, y_test) = get_dataset('ember',True,True)
    model = get_model('Ember_2')
    X_adv = generate_attack_tf(model,X_test,y_test,'EMBER')
    generate_activations(X_adv,y_test,model,'./adversarial/test')
