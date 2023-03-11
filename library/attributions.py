import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Accessor import *
import numpy as np 
from sklearn.model_selection import train_test_split
from attributionUtils import NeuralNetMnist_1,NeuralNetMnist_2,NeuralNetMnist_3, NeuralNetEmber, NeuralNetCuckoo_1, NeuralNetCifar_10
from attributionUtils import attribtuions_to_polarity,predict_torch,compute_accuracy_torch
from attributionUtils import get_attributes,adversarial_detection_set
import random
from attributionUtils import KNN 
from utils import normalize,dispersation_index
from metrics import expD,expE
from Activations import Activations
import math


begning_accessor = Accessor('./begnign/mnist/mnist_1')
adversarial_accessor= Accessor('./adversarial/mnist/FGSM/mnist_1')
ground_truth_accessor = Accessor('./Ground_truth/mnist/mnist_1')
expected_nb_nodes = 420

for prediction in range(10):

    begning_sample_act = begning_accessor.get_label_by_prediction( target_prediction = prediction ,collapse='avg')
    adv_sample_act = adversarial_accessor.get_label_by_prediction( target_prediction = prediction ,collapse='avg')
    gt_sample_act = ground_truth_accessor.get_label_by_prediction(target_prediction = prediction ,collapse='avg')


    '''
        To categorical  :
        [1. 0.] => 0   =>Benign

    '''

    # Prepare dataset with label 1 for adversaral
    X_adv,Y_adv=adversarial_detection_set(adv_sample_act,label = torch.tensor(1),expected_nb_nodes=expected_nb_nodes)

    # Prepare dataset with label 0 for adversaral
    X_ben,Y_ben=adversarial_detection_set(begning_sample_act,label = torch.tensor(0),expected_nb_nodes =expected_nb_nodes)
    X_gt ,Y_gt =adversarial_detection_set(gt_sample_act,label = torch.tensor(0),expected_nb_nodes=expected_nb_nodes)

    
    print(f'Adv : {len(X_adv)}  Y : {len(Y_adv)}')
    print(f'Ben : {len(X_ben)}  Y : {len(Y_ben)}')
    #print(f'GT : {len(X_gt)}  Y : {len(Y_gt)}')
    
    '''
    X = X_adv+ X_ben
    Y = Y_adv+ Y_ben

    Y = to_categorical(Y)
    model = NeuralNetCifar_10()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    X_train, X_test,y_train, y_test = train_test_split(X,Y ,random_state=104, test_size=0.25, shuffle=True)
    print(f'X_test len {len(X_test)}')
    print(f'Y_test len {len(y_test)}')

    accuracy = 1
        #Train Model
    while(accuracy < 83):
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
    
    torch.save(model, './advDetectionModels/torch_Cifar10_3.pt')
    exit()

    '''




    model = torch.load('./advDetectionModels/torch_Mnist_1_FGSM.pt')

    def get_average(X):
        if(len(X)==0) : raise ValueError('Cannot compute average of an empty array')
        return np.average(np.array(X)) /len(X)

    def remove_elements_below_threshhold(attributes,x,theshhold):
        aux = []
        for i in attributes :
            if( abs(i)< abs(theshhold)):
                continue
        aux.append(i)
        print(len(aux))
        return torch.tensor(aux)
    
    def normalize(set):
        #Normalize a set to range [-1,1]
        min_val = min(set)
        max_val = max(set)
        normalized_set = []
        for val in set:
            normalized_val = ((val - min_val) / (max_val - min_val)) * 2 - 1
            normalized_set.append(normalized_val)
        return normalized_set
    def get_items_below_threshhold(attributes,threshhold):
        #given a set S this function returns ths iindices of elements that are below the threshholdin  abs value
        # this is for attribution to remove nodes who's attributions are ireelevant
        aux= []
        for index,x in enumerate(attributes):
            if(abs(x) < abs(threshhold)):
                continue
            aux.append(index)
        return aux
    
    def remove_element_with_indexes(array,indexes):
        #given a set and an array of indexes this funciton remives the all the indexes form the array
        aux =[]
        for i,x in enumerate (array) : 
            if( i in indexes):
                continue
            aux.append(x)
        return torch.Tensor(aux)
 
    def stats(attributes):
        pos =0
        neg= 0
        z = 0
        for i in attributes:
            if(i<0):neg+=1
            if(i>0):pos+=1
            else : z +=0
        
        return pos >neg

    def multiply_attributed_with_input(X,Y):
        s = []
        
        for index,input in enumerate(X):
            label = Y[index]
            prediction = predict_torch(model,input).item()
            # filter to only correct prediction
            if( prediction != label): continue
            attributes = get_attributes(input,model,prediction)           
            # find indexes of attributes that are below the thresshhold
            #indexes_to_remove = get_items_below_threshhold(attributes,0.01)
           
            #remove those indexes from attribtuions and from input { Considered as noise}
            #attributes = remove_element_with_indexes(attributes,indexes_to_remove)
            #input = remove_element_with_indexes(input,indexes_to_remove)
      
            #multiply  attributes with input
           
            attributes = np.absolute(attributes)
            mul = np.multiply(attributes,input)

            s.append(mul)
            printProgressBar(index, len(X), prefix = 'Progress:', suffix = 'Complete', length = 50)
        return s
    
    def number_of_active_nodes(set):
        threshold = 0.0001
        counter = 0
        for i in set : 
            if( i != 0 ):
                counter+=1
        return counter


    X_ben = multiply_attributed_with_input(X_ben,Y_ben)
    X_adv = multiply_attributed_with_input(X_adv,Y_adv)
    X_gt = multiply_attributed_with_input(X_gt,Y_gt)

    X_ben = np.average([number_of_active_nodes(x) for x in X_ben])
    X_adv = np.average([number_of_active_nodes(x) for x in X_adv])
    X_gt = np.average([number_of_active_nodes(x) for x in X_gt])
    print(f' Nb active nodes : ben: {X_ben}  adv : {X_adv} gt: {X_gt}')
    
    #plt.bar(['ben', 'adv', 'gt'] , [X_ben,X_adv,X_gt])
    #plt.show()