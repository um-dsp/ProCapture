from library.generate_activations import generate_activations
from library.utils import get_dataset,get_model,generate_attack_tf
import numpy as np
import sys

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

supported_dataset = ['cifar10' ,'mnist', 'cuckoo','ember'] 
supported_attacks = ['FGSM','CW','PGD',"CKO",'EMBER',None]
pre_trained_models = ['cifar10_1','cuckoo_1','Ember_2','mnist_1','mnist_2','mnist_3']
folder = ['Ground_Truth' , 'Begnign' ,'Adversarial']
def parseArgs():
    args= sys.argv
    dataset = args[1]
    if( dataset not in supported_dataset):
        raise ValueError(f'ProvMl only Supports {supported_dataset}')
        
    model_name = args[2]
    if(model_name not in pre_trained_models ):
        raise ValueError(f'ProvMl only Supports {pre_trained_models}')
        

    folder = args[3]
    if(folder not in folder):
        raise ValueError(f"ProMl save folder options are {folder}")

    # Optional Attack argument, if None no attack will be performed on the input
    if (len(args)==5 ):
        attack = args[4]
        if(attack not in supported_attacks):
          raise ValueError(f'ProvMl only Supports {supported_attacks}')
    else :
        attack = None

  
    if(not attack and folder =="adversarial"):
      raise ValueError('cannot save adversarial checkpoint without Attack Input')

  
    return dataset,attack,model_name,folder


def get_checkpoint_name(dataset,attack,model_name,folder):
    # Generate FoLder Path 
    save_path = folder +'/' +dataset +'/'
    if(attack):
        save_path+= attack +'/'
    save_path += model_name 
    return save_path


if __name__ == "__main__":
   
    dataset,attack,model_name,folder = parseArgs()

    save_path = get_checkpoint_name(dataset,attack,model_name,folder)

    print(f'\n[GEN ACT] Dataset : {dataset} | Model : {model_name} | Attack : {attack} | Checkpoint : {save_path} \n')
    #Use the below  function to generate activations for different datasets/attacks
      # Cifar generation Code


    # Ground Truth -> We Use Train Data 
    # Adersarial | Begnign -> We use Test Data
    (X_train, y_train), (X_test, y_test) = get_dataset(dataset,True)
    if(folder == 'Ground_Truth'):
        X = X_train 
        Y= y_train 
    else : 
        X= X_test 
        Y = y_test


    model = get_model(model_name)
 
    if(attack):
      X = generate_attack_tf(model,X,Y,attack)

    generate_activations(X,Y,model,save_path)


    

# This is some sample code to look at if needed
def sampleCode():

  # Mnist generation Code
    
    (X_train, y_train), (X_test, y_test) = get_dataset('mnist',False,True)
    model = get_model('mnist_1')
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
