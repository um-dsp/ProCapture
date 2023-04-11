from nntplib import GroupInfo
from operator import index
from re import S
from library.utils import generate_attack_tf, get_model
from library.utils import get_dataset
from library.utils import printProgressBar,discretize,scotts_rule
import numpy as np
from library.Accessor import Accessor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error





def  expD(begning_sample_act,adv_sample_act,ground_truth_act,p=None):
    # Purpose of this experiment is to compute the number of nodes active(output>0) in each activation
    #Compute median and average of each Prediction
    #Comapred
    threshhold = 0

    avg_active_node_miss =[]
    for y in adv_sample_act :
        y.set_layer_range(1,float('+inf'))
        avg_active_node_miss.append(y.compute_nb_active_nodes(threshhold)) 

    avg_active_node_hits = []
    for y in begning_sample_act :
        y.set_layer_range(1,float('+inf'))
        avg_active_node_hits.append(y.compute_nb_active_nodes(threshhold))

    gt_nb = []  
    for y in ground_truth_act :
        y.set_layer_range(1,float('+inf'))
        gt_nb.append(y.compute_nb_active_nodes(threshhold)) 


    g = np.average(gt_nb)
    b = np.average(avg_active_node_hits)
    a=np.average(avg_active_node_miss)
    if(p): print(f'prediction {p}')
    print('\n Avg of active nodes   Begnign : %s  Adversarial : %s  Gt : %s ' %( b, a,g))

    return b,a,g


def expE(begning_sample_act,adv_sample_act,ground_truth_act,p=None):
    #purpose of this experiment is to detemine the average weight of activations 
    # and comapre adversarial,begnign and training
    

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
    
    b = sum_b/len(begning_sample_act)
    a = sum_a/len(adv_sample_act)
    g =  sum_g/len(ground_truth_act)

    if(p): print(f'prediction {p}')
    print(' Average activation weights of   Ben : %s  Adv : %s GT : %s '%(b,a,g))

    return b,a,g


def expF(begning_sample_act,adv_sample_act,ground_truth_act,p=None):
    # This experiment tres to find nodes that are always active 

    threshhold = 0
   
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
  
        if(p): print(f'prediction {p}')

        print('number of nodes always active benign : %s'%(len(always_active_nodes_b)))
        print(always_active_nodes_b)
        print('number of nodes always active Adv : %s'%(len(always_active_nodes_adv)))
        print(always_active_nodes_adv)
        print('number of nodes always active GT : %s'%(len(always_active_nodes_gt)))
        print(always_active_nodes_gt)
        

        print('Number of common nodes betwen Always_active_begnign and always_active_gt : %s'%(counter_b))
        print('Numver of common nodes betwen always_active_adv and always_active_gt : %s'% (counter_adv))


def expG( begning_sample_act,adv_sample_act,ground_truth_act,p=None):
    #Purpsoe fo this experiment is to compute the frquency of node sactivations


    
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

    if(p): print(f'prediction {p}')

    b_gt = np.average(np.absolute(begnig_gt))
    a_gt = np.average(np.absolute(adv_gt))
    print('distance between Benign and Gt: %s' %(b_gt))
    print('distance between Adversarial and Gt: %s ' %(a_gt))

    return b_gt, a_gt


def expH(begning_sample_act,adv_sample_act,ground_truth_act,p=None):
    #Dispersation index

    index_adv = 0
    for i in adv_sample_act:
        index_adv += i.dispersation_index() / len(adv_sample_act)

    
    index_ben = 0
    for i in begning_sample_act:
        index_ben += i.dispersation_index() / len(begning_sample_act)
    
  
    index_gt = 0
    for i in ground_truth_act:
        index_gt += i.dispersation_index() / len(ground_truth_act)

    if(p): print(f'prediction {p}')

    print(f'dispersation index Benign : {index_ben} Adv : {index_adv} GT : {index_gt} ')

    return index_ben,index_adv, index_gt 

   
def expI(begning_sample_act,adv_sample_act,ground_truth_act,p=None):

    #COmputes entropy of sampels
    adv_entrpy = 0
    for i in adv_sample_act:
        adv_entrpy += i.compute_entropy() / len(adv_sample_act)

    ben_entropy = 0
    for i in begning_sample_act:
        ben_entropy += i.compute_entropy() / len(begning_sample_act)
    
  
    gt_entropy = 0
    for i in ground_truth_act:
        gt_entropy += i.compute_entropy() / len(ground_truth_act)

    #print('Prediction %s  layer  Dispersation index Gt : %s  Benign : %s  Adv : %s '%(index_gt,index_ben,index_adv))
    if(p): print(f'prediction {p}')
    print(f'Entropy index   Benign : {ben_entropy} Adv : {adv_entrpy} GT : {gt_entropy}')

    return ben_entropy,adv_entrpy,gt_entropy
   

if __name__ == "__main__":

    adversarial_sample = Accessor('./adversarial/mnist/FGSM/mnist_1')
    begning_sample = Accessor('./begnign/mnist/mnist_1')
    ground_truth = Accessor('./Ground_truth/mnist/mnist_1')

    begning_sample_act = begning_sample.get_all(collapse='avg',limit=1000)   
    adv_sample_act = adversarial_sample.get_all(collapse='avg',limit=1000)
    ground_truth_act = ground_truth.get_all(collapse='avg',limit=1000)
    expI(begning_sample_act,adv_sample_act,ground_truth_act)


    exit()
    for i in range(0,10):
        begning_sample_act = begning_sample.get_label_by_prediction(target_prediction= i,collapse='avg',limit=1000)   
        adv_sample_act = adversarial_sample.get_label_by_prediction(target_prediction = i,collapse='avg',limit=1000)
        ground_truth_act = ground_truth.get_label_by_prediction(target_prediction = i,collapse='avg',limit=1000)
        expI(begning_sample_act,adv_sample_act,ground_truth_act,i)

