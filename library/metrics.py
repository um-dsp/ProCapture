import numpy as np
from library.Accessor import Accessor



def avg_act_diff(act1,act2,nb_sample=1000):
    '''This function computes the difference between activation weights of two sets 
    of activations across all samples 'nb_samples'''
    
    for i in range(nb_sample):
        if i==0:
            diff=abs(np.subtract(np.array(act1[i].flatten()),np.array(act2[i].flatten())))
        else:
            diff = diff+abs(np.subtract(np.array(act1[i].flatten()),np.array(act2[i].flatten())))
    #print(diff)
    return (diff/nb_sample).tolist()

def compare(act1,act2,nb_sample=1000):
    
    '''This function computes the similarity of two sets 
    of activations across all samples 'nb_samples'.'''
    
    avg_sim=0
    nb_nodes=act1[0].get_nb_nodes()
    for i in range(nb_sample):
        sim=0
        
        #if act1[i].get_activations_set()== act2[i].get_activations_set():
            #print('{}% of the nodes of sample {} gave similar activations'.format(100,i))
        #    avg_sim+=1
        try:
            for j in range(nb_nodes):
                if act1[i].flatten()[j] == act2[i].flatten()[j]:
                    sim+=1
            #print('{}% of the nodes of sample {} gave similar activations'.format(100*sim/nb_nodes,i))
            avg_sim+=sim/nb_nodes
        except IndexError:
            pass
        
    return 100*avg_sim/nb_sample

def  expD(act):
    # Purpose of this experiment is to compute the number of nodes active(output>0) in each activation
    #Compute median and average of each Prediction
    #Comapred
    threshhold = 0
    
    avg_active_node =[]  
    for y in act:
        y.set_layer_range(1,float('+inf'))
        avg_active_node.append(y.compute_nb_active_nodes(threshhold)) 
    
    return np.average(avg_active_node)
    



def expE(act):
    #purpose of this experiment is to detemine the average weight of activations 
    # and comapre adversarial,begnign and training
    

    summ = 0
    for x in act:
        summ += x.get_average_weight() 
    
    
    return summ/len(act)


def expF(act):
    # This experiment tres to find nodes that are always active 

    threshold = 0
    
    # Intitialize a list of always active nodes
    
    always_active_nodes = list(range(act[0].get_nb_nodes()))
    
    
    # Look for cases where a node is deactivated and remove it
    for x in act :
        unactive_nodes = []
        for j,y in enumerate(x.flatten()):
            if(type(y) == type([])):continue
            if(abs(y)==threshold):
                unactive_nodes.append(j)
        
        for node in unactive_nodes:
            if node in always_active_nodes:
                always_active_nodes.remove(node)
                
        if(len(always_active_nodes) == 0 ):
            break
        
        
    return always_active_nodes

def expG(act):
    #Purpsoe fo this experiment is to compute the frquency of node sactivations


    threshhold = 0

    nb_nodes_in_model = act[0].get_nb_nodes()
    
    frequency = [0]*nb_nodes_in_model
    #indices=list(range(nb_nodes_in_model))
        


    for i in act:
        #i.set_layer_range(l,l)
        flat = i.flatten()
        if(len(flat) !=nb_nodes_in_model ) : continue
     
        for ind,j in enumerate(flat):
            if(type(j) == type([])): continue

            if( abs(j)>threshhold):
                frequency[ind] +=1 / len(act)
    
    return frequency


def expH(act):
    #Dispersation index

    ind = 0
    for i in act:
        ind += i.dispersation_index()

    
    return ind/len(act)


   
def expI(act):

    #Computes entropy of samples
    entropy = 0
    for i in act:
        entropy += i.compute_entropy() / len(act)
    
    
    
    
    
    return entropy