from operator import contains
from Activations import Activations
import os
import pandas as pd
import numpy as np
from keras import backend as K
from utils import *
class Loader :

    # for all class attack param selects whether to get adersarial activations or begnign activations


    def __init__(self,adv_folder,begnign_folder,dataset):
        self.adv = adv_folder
        self.begnign = begnign_folder
        self.dataset = dataset

    
   
    #get activations by label and prediction [get activations of an input predicted 0 and labeled 0]
    def get_instance_by_label_prediction(self,label,prediction,attack):        
        folder = get_folder_name(attack)
        for filename in os.listdir(folder):
            f = os.path.join(folder, filename)
            # checking if it is a file
            if filename.startswith(str(label) +"_"+ str(prediction)):
                index = filename[filename.index("-")+1:filename.index(".")] 
                activations_set = pd.read_csv(f)
                print('Loaded Activations of image labeled %s predicted %s under attack %s' % (label,prediction,attack))
                return  Activations(index,label,prediction,activations_set,attack)

  

    #gets activations for a given class [0,1,...]
    def get_by_label(self,label,attack):
        folder = get_folder_name(attack)
        container = []
        for filename in os.listdir(folder):
            f = os.path.join(folder, filename)
            if filename.startswith(str(label)):
                index = filename[filename.index("-")+1:filename.index(".")] 
                predicted = filename[filename.index("_"):filename.index("-")] 
                activations_set = pd.read_csv(f)
                container.append(Activations(index,label,predicted,activations_set,None))
        if( len(container) ==0):
            print('No File was found for label %s'%(label))
            raise Exception()
        print('Loaded '+str(len(container))+' Activations for Label : '+str(label))
        return container
    
    #gets activations by input index in dataset
    def get_instance_by_index(self,index,attack):
        folder = get_folder_name(attack)
        container = []
        for filename in os.listdir(folder):
            f = os.path.join(folder, filename)
            i = filename[filename.index("-")+1:filename.index(".")] 
            if (index ==i) :
                predicted = filename[filename.index("_"):filename.index("-")] 
                label = filename[0:filename.index("_")] 
                activations_set = pd.read_csv(f)
                container.append(Activations(index,label,predicted,activations_set,None))
                print("Loaded instance indexed %s labeled %s predicted %s under "%(index,label,predicted,attack))
        if( len(folder) == 0) :
            raise Exception("No file found for index %s" ,index)
        return container
        


    #Generates acivations for a given model and input and saves in corresponding folder
    def generate_and_save_activations(self,model,input,index,label,attack):
        ac =  get_layers_activations(model,input)
        input = np.reshape(input,(-1,28,28))
        prediction =np.argmax(model.predict(input,verbose=0)[0])
        activations = [item for sublist in ac for item in sublist]
        a = Activations(index,label,prediction,activations,attack)
        a.save_csv()
        return a


    #implement get adv and begnign per sammple
    # IMPLEMENT GET smaller dataset containing 1,2,... or n classes only
