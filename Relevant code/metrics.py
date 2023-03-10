from nntplib import GroupInfo
from operator import index
from re import S
from utils import generate_attack_tf, get_model
from utils import get_dataset
from utils import printProgressBar
import numpy as np
from Accessor import Accessor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error





adversarial_sample = Accessor('./adversarial/ember/EMBER/ember_2')
begning_sample = Accessor('./begnign/ember/ember_2')
ground_truth = Accessor('./Ground_truth/ember/ember_2')

    

def expD(p):
    # Purpose of this experiment is to compute the number of nodes active(output>0) in each activation
    #Compute median and average of each Prediction
    #Comapred

    begning_sample_act = begning_sample.get_label_by_prediction(target_prediction= 1,collapse='avg',verbose = 0,limit=1000)   
    adv_sample_act = adversarial_sample.get_label_by_prediction(target_prediction = 1,collapse='avg',verbose = 0,limit=1000)
    ground_truth_act = ground_truth.get_label_by_prediction(target_prediction = 1,collapse='avg',verbose =0 ,limit=1000)
    threshhold = 0

    

    avg_active_node_miss =[]
    for y in adv_sample_act :
        avg_active_node_miss.append(y.compute_nb_active_nodes(threshhold)) 

    avg_active_node_hits = []
    for y in begning_sample_act :
        avg_active_node_hits.append(y.compute_nb_active_nodes(threshhold))

    gt_nb = []  
    for y in ground_truth_act :
        gt_nb.append(y.compute_nb_active_nodes(threshhold)) 

    print('Avg of active nodes for prediction : %s  Gt : %s   Begnign : %s  Adversarial : %s' %(p,np.average(gt_nb),
    np.average(avg_active_node_hits),
    np.average(avg_active_node_miss)))



def expE(i):
    #purpose of this experiment is to detemine the average weight of activations 
    # and comapre adversarial,begnign and training
    

    begning_sample_act = begning_sample.get_label_by_prediction(target_prediction=i)   
    adv_sample_act = adversarial_sample.get_label_by_prediction(target_prediction=i)
    ground_truth_act = ground_truth.get_label_by_prediction(target_prediction=i)

    print(len(begning_sample_act))
    print(len(adv_sample_act))
    print(len(ground_truth_act))



    sum_b = 0
    for x in begning_sample_act:
        #x.set_layer_range(l,l)
        sum_b += x.get_average_weight() 
    

    sum_a = 0
    for x in adv_sample_act:
        #x.set_layer_range(l,l)
        sum_a += x.get_average_weight()

    
    sum_g = 0
    for x in ground_truth_act:
        #x.set_layer_range(l,l)
        sum_g += x.get_average_weight()


    print('Prediction : %s  Average activation weights of Ben : %s  Adv : %s  GT : %s'%
    (i,sum_b/len(begning_sample_act)
    ,sum_a/len(adv_sample_act),
    sum_g/len(ground_truth_act)))

    return


def expF(p):
    # This experiment tres to find nodes that are always active 

    threshhold = 0

    begning_sample_act = begning_sample.get_label_by_prediction(p,collapse='avg')
    adv_sample_act = adversarial_sample.get_label_by_prediction(p,collapse='avg')
    ground_truth_act = ground_truth.get_label_by_prediction(p,collapse='avg')
    #Removing outliers
   
   
    counter = 0



    always_active_nodes_b = []
    for i in range(0,begning_sample_act[0].get_nb_nodes()):
        always_active_nodes_b.append(i)

    for i,x in enumerate(begning_sample_act) :
        active_nodes = []
        for j,y in enumerate(x.flatten()):
            if(type(y) == type([])):continue
            if(y>threshhold):
                active_nodes.append(j)
        pop_counter = 0
        for index,g in enumerate(always_active_nodes_b):
            if(g not in active_nodes):
                always_active_nodes_b.pop(index)
                pop_counter += 1
        if(len(always_active_nodes_b) == 0 ):
            break



    always_active_nodes_adv = []
    for i in range(0,adv_sample_act[0].get_nb_nodes()):
        always_active_nodes_adv.append(i)
    for i,x in enumerate(adv_sample_act) :
        active_nodes = []
        if(type(y) == type([])):continue

        for j,y in enumerate(x.flatten()):
            if(y>threshhold):
                active_nodes.append(j)
        pop_counter = 0
        for index,g in enumerate(always_active_nodes_adv):
            if(g not in active_nodes):
                always_active_nodes_adv.pop(index)
                pop_counter += 1
        if(len(always_active_nodes_adv) == 0 ):
            break
    

    always_active_nodes_gt = []
    for i in range(0,ground_truth_act[0].get_nb_nodes()):
        always_active_nodes_gt.append(i)
    for i,x in enumerate(ground_truth_act) :
        active_nodes = []
        for j,y in enumerate(x.flatten()):
            if(type(y) == type([])):continue
            if(y>threshhold):
                active_nodes.append(j)
        pop_counter = 0
        for index,g in enumerate(always_active_nodes_gt):
            if(g not in active_nodes):
                always_active_nodes_gt.pop(index)
                pop_counter += 1
        if(len(always_active_nodes_gt) == 0 ):
            break
        counter_b = 0
        for i in always_active_nodes_b:
            if( i in always_active_nodes_gt):
                counter_b+=1
        counter_adv = 0 
        for i in always_active_nodes_adv:
            if(i in always_active_nodes_gt):
                counter_adv +=1
        print(f'Prediction {p} ')
        print('number of nodes always active benign : %s'%(len(always_active_nodes_b)))
        print(always_active_nodes_b)
        print('number of nodes always active Adv : %s'%(len(always_active_nodes_adv)))
        print(always_active_nodes_adv)
        print('number of nodes always active GT : %s'%(len(always_active_nodes_gt)))
        print(always_active_nodes_gt)
        
        
        print('Number of common nodes betwen Always_active_begnign and always_active_gt : %s'%(counter_b))
        print('Numver of common nodes betwen always_active_adv and always_active_gt : %s'% (counter_adv))


def expG( p):
    #Purpsoe fo this experiment is to compute the frquency of node sactivations

    begning_sample_act = begning_sample.get_label_by_prediction(target_prediction=p)
    adv_sample_act = adversarial_sample.get_label_by_prediction(target_prediction=p)
    ground_truth_act = ground_truth.get_label_by_prediction(target_prediction=p)


    
    frequency_gt = []
    frequency_be = []
    frequency_adv = []

    threshhold = 0

    nb_nodes_in_model = begning_sample_act[0].get_nb_nodes()

    #compute frequencies , each array will hold the percentage of activations of all nodes len(arra) = nb_nodes
    for i in range(0,nb_nodes_in_model+1):
        frequency_gt.append(0)
        frequency_be.append(0)
        frequency_adv.append(0)


    for i in ground_truth_act:
        #i.set_layer_range(l,l)
        flat = i.flatten()
        if(len(flat) !=nb_nodes_in_model ) : continue
     
        for index,j in enumerate(flat):
            if(type(j) == type([])): continue

            if( j>threshhold):
                frequency_gt[index] +=1 / len(ground_truth_act)



    for i in begning_sample_act:
        #i.set_layer_range(l,l)
        flat = i.flatten()
        if(len(flat) !=nb_nodes_in_model ) : continue

        for index,j in enumerate(flat):
            if(type(j) == type([])): continue
            if( j>threshhold):
                frequency_be[index]+=1 /len(begning_sample_act)



    for i in adv_sample_act:
        #i.set_layer_range(l,l)
        flat = i.flatten()
        if(len(flat) !=nb_nodes_in_model ) : continue
    
        for index,j in enumerate(flat):
            if(type(j) == type([])): continue
            if( j>threshhold):
                frequency_adv[index]+=1 /len(adv_sample_act)


 
    #compute distance between frequencies b-gt  adv-gt 
    begnig_gt = np.array(frequency_be) - np.array(frequency_gt)
    adv_gt = np.array(frequency_adv) -np.array(frequency_gt)

    print(f'Prediction {p}')
    print('distance between Benign and Gt: %s' %(np.average(np.absolute(begnig_gt))))
    print('distance between Adversarial and Gt: %s ' %(np.average(np.absolute(adv_gt))))

    return 


def expH(p):
    #Dispersation index
  
    begning_sample_act = begning_sample.get_label_by_prediction(target_prediction=p)
    adv_sample_act = adversarial_sample.get_label_by_prediction(target_prediction=p )
    ground_truth_act = ground_truth.get_label_by_prediction(target_prediction=p )



    index_adv = 0
    for i in adv_sample_act:
        index_adv += i.dispersation_index() / len(adv_sample_act)

    
    index_ben = 0
    for i in begning_sample_act:
        index_ben += i.dispersation_index() / len(begning_sample_act)
    
  
    index_gt = 0
    for i in ground_truth_act:
        index_gt += i.dispersation_index() / len(ground_truth_act)

    #print('Prediction %s  layer  Dispersation index Gt : %s  Benign : %s  Adv : %s '%(index_gt,index_ben,index_adv))
    print(f'prediction {p} layer : {0} dispersation index  GT : {index_gt} Benign : {index_ben} Adv : {index_adv}')


   
   
   

if __name__ == "__main__":
    for i in range(2):
        expG(i)

