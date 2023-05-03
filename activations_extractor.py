from library.utils import get_dataset,get_model,generate_attack_tf, printProgressBar
from library.Activations import Activations
from library.train import evaluate

import numpy as np
import sys
from batchup import data_source
import pandas as pd
from keras import backend as K
#from keras.utils import to_categorical
import tensorflow as tf

'''
get_dataset() in utils would return a dataset of your choice, 
Datasets implemented are mnist , cifar , ember and cuckoo,
the function will returna partitioned (X_train, y_train), (X_test, y_test) set for training and testing set
Other params of this function are  Normalization and to_categorical which will be executed if their flag is set to True



get_model() will retreive a model of your choice, the models are in ./models folder
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
supported_attacks = ['FGSM','CW','PGD',"CKO",'EMBER_att',None]
pre_trained_models = ['cifar10_1','cuckoo_1','Ember_2','mnist_1','mnist_2','mnist_3']
folder = ['Ground_Truth' , 'Benign' ,'Adversarial']
def parseArgs():
    args= sys.argv
    dataset = args[1]
    if( dataset not in supported_dataset):
        raise ValueError(f'ProvML only Supports {supported_dataset}')
        
    model_name = args[2]
    if(model_name not in pre_trained_models ):
        raise ValueError(f'ProvML only Supports {pre_trained_models}')
        

    folder = args[3]
    if(folder not in folder):
        raise ValueError(f"ProvML save folder options are {folder}")

    # Optional Attack argument, if None no attack will be performed on the input
    if (len(args)==5 ):
        attack = args[4]
        if(attack not in supported_attacks):
          raise ValueError(f'ProvML only Supports {supported_attacks}')
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




def select_only_one_label (X,Y,label):
    aux = []
    for m,n in enumerate(Y):
        if(np.argmax(n)== label):
            aux.append(X[m])
    return np.array(aux)

def is_wrong_prediction (model,x,y,i):
    pred = np.argmax(model.predict(x,verbose =0))
    return pred == y


# return the activations of each mlayer of the mdel  
def get_layers_activations(model,x):
    inp = model.input                                           
    outputs = [layer.output for layer in model.layers]          
    functors = [K.function([inp], [out]) for out in outputs]  
    layer_outs = [func([x]) for func in functors]
    return layer_outs

def generate_activations(X,Y,model,file_path):

    counter = 1
    correct_predictions = 0
    for i,x in enumerate(X):
        # Catgorized label to label
        if(isinstance(Y, pd.DataFrame)) : y = Y.iloc[i]
        else : y = Y[i]


        #Reshape needed for K backend logits extractions
        x = np.expand_dims(x,axis= 0)
        #generate and save activations nd return if sucessfull prediction or not
        if(generate_and_save_activations(model,x,i,y,file_path)):
            '''accuracy should be 100% for benign activations and 0% for adversarial ones'''
            correct_predictions+=1
        #break
        #Print the accuracy so far to monitor hidden layer extraction
        if(counter %50 ==0):
            print(f'accuracy so far : {correct_predictions/counter*100} %')

        printProgressBar(i + 1, X.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
        counter+=1
    print("Generated and saved set activations dataset to %s " %(file_path))


#Generates acivations for a given model and input and saves in corresponding folder
def generate_and_save_activations(model,x,index,label,folder_name):
    '''We generate activations on for evasive adversarial samples
    and correctly predicted benign samples'''
    
    ac =  get_layers_activations(model,x)
    #print(ac)
    #print(len(ac))
    prediction =np.argmax(model.predict(x,verbose=0)[0])
    if(label.shape[0]!= 1):
        label =np.argmax(label)
    ## For adversarial activation extraction we are interested only in evasive samples
    if any(adv in folder_name for adv in ['Adversarial','FGSM','PGD','CW','CKO','EMBER_att']):
    
        if prediction==label: #skip extraction
            #print("skipping extraction for unevasive sample")
            return False
    else:
        if prediction!=label: #skip extraction
            #print("skipping extraction for naturally evasive sample")
            return True
        
    
    #extracting arrays of each layer's node
    activations = [item for sublist in ac for item in sublist]
    #for arr in activations:
    #     print(arr.shape)
    #print(activations)
    #print(len(activations))
    #print(np.array(activations[0][0]).shape)

    #For string label translated to dummy categories need to format label to string to put in file name
    '''
    if(isinstance(label,pd.Series)):
       label = label['Benign'].astype(str) + label['Malware'].astype(str)
    '''
    lst= []
    for i in activations : 
        arr = np.array(i)
        if(len(arr.shape) ==4):
            arr = np.moveaxis(arr, [0,1,2,3], [3,2,1,0])
        if(len(arr.shape)==2):
            arr = np.moveaxis(arr, [0,1], [1,0])
        arr = np.squeeze(arr)
        lst.append(arr)
   
    #print(len(lst))
    #print(lst)
    
    #for arr in lst:
    #     print(arr.shape)
    

    a = Activations(index,label,prediction,lst)
    a.save_cnn(lst,folder_name)
    return label == prediction 


if __name__ == "__main__":
   
    dataset,attack,model_name,folder = parseArgs()

    save_path = get_checkpoint_name(dataset,attack,model_name,folder)

    print(f'\n[GEN ACT] Dataset : {dataset} | Model : {model_name} | Attack : {attack} | Checkpoint : {save_path} \n')
    #Use the below  function to generate activations for different datasets/attacks
      # Cifar generation Code


    # Ground Truth -> We Use Train Data 
    # Adersarial | Benign -> We use Test Data
    (X_train, y_train), (X_test, y_test) = get_dataset(dataset,True)
    if(folder == 'Ground_Truth'):
        X = X_train 
        Y= y_train 
    else : 
        X= X_test 
        Y = y_test


    model = get_model(model_name)
    evaluate(model,X,tf.convert_to_tensor(Y))
    
        
    if(attack):
        X_adv=X.copy()
        ds = data_source.ArrayDataSource([X, Y])
        i=0
        batch_size=200
        for (batch_X, batch_y) in ds.batch_iterator(batch_size=batch_size):
            print('Generating {} from sample {} to {}'.format(attack,i*batch_size,(i+1)*batch_size-1))
            X_adv[i*batch_size:(i+1)*batch_size] = generate_attack_tf(model,batch_X, batch_y,attack)
            #evaluate(model,X_adv[i*batch_size:(i+1)*batch_size],tf.convert_to_tensor(batch_y))
            i+=1
        X=X_adv
        #evaluate(model,X_adv,tf.convert_to_tensor(Y))
    #print(tf.math.argmax(tf.convert_to_tensor(Y),axis=1))
    #print(tf.math.argmax(model(X),axis=1))
    evaluate(model,X,tf.convert_to_tensor(Y))
    #print('accuracy on all data: ',compute_accuracy_tf(model,X,Y))
    generate_activations(X,Y,model,save_path)