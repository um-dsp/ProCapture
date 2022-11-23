import matplotlib.pyplot as plt
import os
from Accessor import Accessor
import pandas as pd
from Activations import *
class Validator : 

       
    def check_nb_layers(self):
        for a in self.activations :
            if(len(self.activations.flatten())!=self.nb_nodes):
                raise Exception("Faulty activation generated")

    def filter_on_layers(self):
        for i,a in enumerate(self.activations) :
            if(len(self.activations.flatten())!=self.nb_nodes):
                self.activations.pop(i)
        return self.activations


    def accuracy_from_file (self,folder):
        trues= 0
        counter = 0
        for filename in os.listdir(folder):
            counter+=1
            f = os.path.join(folder, filename)
            predicted = filename[filename.index("_")+1:filename.index("-")] 

            #Differ in cocku and other dataset
            label = filename[0:filename.index("_")] 
            if(predicted == label):
                trues+=1
        print('Counter : %s' %(counter))
        print('Accuracy : %s ' %(trues/counter *100))
        return trues
    
    def detect_inconsistencies(self,folder,expected):
        count =[]
        for filename in os.listdir(folder):
            f = os.path.join(folder, filename)
            i = int(filename[filename.index("-")+1:filename.index(".")] )
            activations_set = self.parse_csv_to_set(pd.read_csv(f))
            print(activations_set)
            exit()
            if(len(activations_set) != expected):
                count.append(i)
        print('Number of Layer inconsistencies : %s' %(len(count)))
        if(len(count) == 0 ) :
            raise Exception('No inconsistency detected')
        return count

    def parse_csv_to_set(self,file_content):
        set = []
        for col in file_content:
            set.append(file_content[col].dropna())
        
        return set
  

if __name__ == "__main__":
    expected_nb_layers = 4
    v = Validator()
    v.accuracy_from_file('./adversarial/cifar10/FGSM/cifar10_1')





        

    