from re import I
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import get_folder_name

class Activations : 
  
    def __init__(self,index,label,prediction,activation_set,attack):
        self.index= index
        self.label = label
        self.prediction =prediction
        self.activations_set = activation_set
        self.attack=attack
    
    

    def save_csv(self,dataset):
        if(not self.activations_set):
            raise ValueError('Activations parameters are  undefined, cannot save')
 
        DF = pd.DataFrame(self.activations_set[0][0],columns=["Layer_0"]) 
        for i,out in enumerate(self.activations_set):
            if (i == 0 ):
                continue
            DF["Layer_"+ str(i)] = pd.DataFrame(self.activations_set[i][0])
            # save the dataframe as a csv file
            #Fill in layer 1 information (node values) in a column in the CSV 
        filename = str(self.label) + "_"+ str(self.prediction) +"-"+str(self.index)

        folder_name = get_folder_name(self.attack,dataset)
       
        DF.to_csv(folder_name +"/"+filename +".csv", index=False)


    
    def as_dot(self,start_layer =1,end_layer=3):
        file1 = open("graph.txt","w")
        L = ["digraph G {\n", "rankdir = LR;\n", "splines=line;\n","ranksep= 1.4;\n","node [fixedsize=true, label=\"\"];\n"]  
        file1.writelines(L)
        L=["subgraph cluster_0 {\n","color=white;\n","node [style=solid, shape=circle];\n"]
        file1.writelines(L)
        for i,activation in enumerate(self.activations_set):
            if( i< start_layer or i>end_layer):
                continue
            for r in range (0, (activation[0]).size):
                file1.write("L"+ str(i)+"N")
                file1.write(str(r))
                file1.write("[label=N")
                file1.write(str(r))
                file1.write("]")
                if (activation[0].item(r))!=0:
                    file1.write("[shape=circle, style=filled, fillcolor=green]")
                file1.write("\nlabel=\"Layer "+str(i)+"\"")    
                file1.write(" ")

        file1.write("\n}")

        index = start_layer
        while(index != end_layer):
            for r in range (0, (self.activations_set[index][0]).size):
            
                for j in range (0, (self.activations_set[index+1][0]).size):
                    file1.write("L"+str(index)+"N")
                    file1.write(str(r))
                    file1.write("->")
                    file1.write("L"+str(index+1)+"N")
                    file1.write(str(j))
                    file1.write("\n")
            print(index)
            index+=1
        
        file1.write("}")
        file1.close() #to change file access modes
        os.system("dot -Tpng -O graph.txt")
        print('Generated graph dot code and saved in graph.txt, can see graph in graph.txt.png')


            

   
    def print(self):
        print('label :', self.label)
        print('prediction :' , self.prediction)
        print('index :',self.index)
        print('attack :',self.attack)

  
   


     


        
