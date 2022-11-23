import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


class Activations : 
  
    def __init__(self,index,label,prediction,activation_set):
        self.index= index
        self.label = label
        self.prediction =prediction
        self.activations_set = activation_set
        # start and end  layer state what subset of layers in activations to consider
        self.start_layer = 0
        self.end_layer = 100
        self.is_clipped = False
     
    def get_activations_set(self):
        if(self.is_clipped):
            return self.activations_set[self.start_layer:self.end_layer+1]
        return self.activations_set  
    def get_label(self):
        return self.label

    def get_nb_layers(self):
        return len(self.activations_set)
    #Saves activations a csv in corresponding folder
    def save_csv(self,folder_name):
        if(not self.activations_set):
            raise ValueError('Activations parameters are  undefined, cannot save')
 
        DF = pd.DataFrame(self.activations_set[0],columns=["Layer_0"]) 
        for i,out in enumerate(self.activations_set):
            if (i == 0 ):
                continue
            DF["Layer_"+ str(i)] = pd.DataFrame(self.activations_set[i])
            # save the dataframe as a csv file
            #Fill in layer 1 information (node values) in a column in the CSV 
        filename = str(self.label) + "_"+ str(self.prediction) +"-"+str(self.index)
      
        DF.to_csv(folder_name +"/"+filename +".csv", index=False)

    def save_cnn(self,activations_list,folder_name):
        layer = 0
        filename = str(self.label) + "_"+ str(self.prediction) +"-"+str(self.index)+'.txt'
        with open(folder_name+'/'+filename, 'a') as out:
            for i in activations_list :
                out.write(f'Layer :{layer}; \n')
                for index,node in enumerate(i) :
                    out.write(f'Node :{index}; \n')
                    charac = np.array(node).flatten()
                    out.write(np.array2string(charac,max_line_width= float('+inf'),separator=','))
                    out.write('\n')
                layer+=1
            out.close()


    #Generates dot file and saves it in dfault folder  
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

     # returns actication set as binary 
    def get_binary(self,threshhold):
        aux = []
        for i,s in enumerate(self.activations_set) :
            aux.append([])
            for a in s :
                if a >threshhold :
                    aux[i].append(1)
                else :
                    aux[i].append(0)
        return aux

    def hamming_index (self,reference_activations,threshhold) :
        #given two binary activations set computes how simialr they are
        # ; how any bits need ti be changed to transform one to the other
        binar_actiovation_set = self.get_binary(threshhold)
        binary_reference_set = reference_activations.get_binary(threshhold)
        index = 0
        for i,x in enumerate(binar_actiovation_set):
            if( i <self.start_layer or i>self.end_layer):
                continue
            
            for j,y in enumerate(x) :
                if(y != binary_reference_set[i][j]):
                    index += 1
            
        return index
         
    def print(self):
        print('label : %s prediction : %s index :%s' %(self.label,self.prediction,self.index))
    
    def compute_nb_active_nodes(self,threshhold):
        nb = 0
        layer_count = 0
        for i,x in enumerate(self.activations_set):
            if(i<self.start_layer or i>self.end_layer):
                continue
            layer_count+=1
            for j,y in enumerate(x):
                if y > threshhold:
                    nb +=1
        return nb




    def get_nb_nodes(self):
        return (len(self.flatten()))

    def flatten(self):
        if(self.is_clipped):
            slice_list = self.activations_set[self.start_layer:self.end_layer+1]
        else :
            slice_list = self.activations_set
        flat_list = [item for sublist in slice_list for item in sublist]
        #print('Flattened List to %s elements' %(len(flat_list)))
        return flat_list

    def set_layer_range(self,start_layer,end_layer):
        self.start_layer =start_layer
        self.end_layer = end_layer
        self.is_clipped = True
        return self

    def get_truth_value(self):
        return self.prediction == self.label

    def get_average_weight(self,nonZero = True):
        res = 0
        for i,x in enumerate(self.activations_set):
            if( i <self.start_layer or i > self.end_layer):
                continue
            res+= np.average(np.array(x))
        
        if(self.is_clipped):
            res = res /(self.end_layer -self.start_layer+1)
        else :
            res = res / len(self.activations_set)
        return res 
    
    def get_nb_active_nodes (self,threshhold):
        count = 0
        for i,x in enumerate(self.activations_set):
            if(i<self.start_layer or i>self.end_layer):
                continue
            for j in x :
                if(j>threshhold):
                    count+=1
        return count 

    
    
    def dispersation_index(self):
        if(self.is_clipped):
            a = self.activations_set[self.start_layer:self.end_layer+1]
            a = [item for sublist in a for item in sublist]
        else :
             a = self.flatten()
       
        return np.var(a) * np.var(a) /np.mean(a)    
   


    def plot(self,color ="green",label=''):


        axis = np.arange(len(self.flatten()))

        plt.figure(figsize=(10, 4))
        plt.xlabel("Nodes")
        plt.ylabel("Activaiton weight")       
        plt.scatter(axis, self.flatten(), s=7,color =color)
        plt.title(label=label)

        #plot verticacl line to tresprresents layers seperatio
        prev = 0
        for i in  self.activations_set[self.start_layer:self.end_layer]:
            plt.axvline(x = len(i)+prev, color = 'b', label = 'Layer')
            prev = prev+len(i)
        plt.show()
       


        
