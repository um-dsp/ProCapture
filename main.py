from nntplib import GroupInfo
from operator import index
from re import S
from utils import generate_attack_tf, get_model
from utils import get_dataset
from utils import printProgressBar,get_input_from_activation
import numpy as np
from Accessor import Accessor
import matplotlib.pyplot as plt

model = get_model("mnist_1")

(X_train, Y_train), (X_test, Y_test) = get_dataset('mnist',True,False,True)


'''
mnist_1  middle are index 1,2
mnist_2  0,1
mnist_3   0,1
cuckoo 1,2,3
'''

def expA ():
    #In this expirament we evaluate average of hamming index of adversarial test data relative to train data
    #(Ground truth) and comapre it to the same result of begnign test data
    # Purpose is to evaluate if the begnign activations are 'closer' to train activations then adversarial activations
   
    user_inp = 1
   
    adversarial_sample = Accessor('./adversarial/mnist/PGD/mnist_1',verbose = 0)
    begning_sample = Accessor('./begnign/mnist/mnist_1')
    ground_truth = Accessor('./Ground_truth/mnist/mnist_1')

    class_activations_adv = adversarial_sample.get_label_by_prediction(user_inp)
    class_activations_begnign = begning_sample.get_label_by_prediction(user_inp)
    ground_truth_activations = ground_truth.get_label_by_prediction(user_inp)

    limit = 7000
    
    indexes = [[]]
    for j,y in enumerate(class_activations_adv) :
        for i,x in enumerate(ground_truth_activations) :
            if i >limit:
                break
            indexes[j].append(y.hamming_index(x,0))
        printProgressBar(j, len(class_activations_adv), prefix = 'Progress:', suffix = 'Complete', length = 50)
        if(j<len(class_activations_adv)-1):
            indexes.append([])
    arr = np.array(indexes)
    np.savetxt('./hamilton_indexes/adv.csv',arr,fmt = '%s')
    
    indexes = [[]]
    for j,y in enumerate(class_activations_begnign) :
        for i,x in enumerate(ground_truth_activations) :
            if i> limit : 
                break
            indexes[j].append(y.hamming_index(x,0))
        printProgressBar(j, len(class_activations_begnign), prefix = 'Progress:', suffix = 'Complete', length = 50)
        if(j<len(class_activations_begnign)-1):
                    indexes.append([])        
    arr = np.array(indexes)
    np.savetxt('./hamilton_indexes/begnign.csv',arr,fmt='%s')       



 
   


def expD(i):
    # Purpose of this experiment is to compute the number of nodes active(output>0) in each activation
    #Compute median and average of each Prediction
    #Comapred
    targer_prediction = i
    adversarial_sample = Accessor('./adversarial/cifar10/FGSM/cifar10_1')
    begning_sample = Accessor('./begnign/cifar10/cifar10_1')
    ground_truth = Accessor('./Ground_truth/cifar10/cifar10_1')

    begning_sample_act = begning_sample.get_label_by_prediction(targer_prediction,collapse='avg')   
    adv_sample_act = adversarial_sample.get_label_by_prediction(targer_prediction,collapse='avg')
    ground_truth_act = ground_truth.get_label_by_prediction(targer_prediction,collapse='avg')
    threshhold = 0

    '''
    expected_nb_of_layers= 4
    # Clean data inconsistencies : 
    for x,y,z in zip(begning_sample_act,ground_truth_act,adv_sample_act):
        if(x.get_nb_layers != expected_nb_of_layers or y.get_nb_layers != expected_nb_of_layers  or z.get_nb_layers != expected_nb_of_layers):
            raise Exception('Inconsistant Activaitons length')
    '''
    
    #layers_range = [0,1]
    #for l in layers_range:

    avg_active_node_miss =0
    for y in adv_sample_act :
        #y.set_layer_range(l,l)
        avg_active_node_miss += y.compute_nb_active_nodes(threshhold) / len(adv_sample_act)




    avg_active_node_hits = 0
    for y in begning_sample_act :
        #y.set_layer_range(l,l)
        avg_active_node_hits += y.compute_nb_active_nodes(threshhold) / len(begning_sample_act)



    gt_nb = 0  
    for y in ground_truth_act :
        #y.set_layer_range(l,l)
        #y.set_layer_range(1,2)
        gt_nb += y.compute_nb_active_nodes(threshhold) / len(ground_truth_act)
    
    print('Avg of active nodes for prediction : %s layer: %s Gt : %s   Begnign : %s  Adversarial : %s' %(targer_prediction,1,gt_nb,avg_active_node_hits,avg_active_node_miss))



def expE(i):
    #purpose of this experiment is to detemine the average weight of activations 
    # and comapre adversarial,begnign and training
    targer_prediction = i
    adversarial_sample = Accessor('./adversarial/cifar10/FGSM/cifar10_1')
    begning_sample = Accessor('./begnign/cifar10/cifar10_1')
    ground_truth = Accessor('./Ground_truth/cifar10/cifar10_1')


    begning_sample_act = begning_sample.get_label_by_prediction(targer_prediction)
    adv_sample_act = adversarial_sample.get_label_by_prediction(targer_prediction)
    ground_truth_act = ground_truth.get_label_by_prediction(targer_prediction)


    '''
    sample = begning_sample_act[0]
    sample.set_layer_range(0,5)
    nb_nodes = sample.get_total_number_of_node()
    print('Detected %s Node' %(nb_nodes))
    '''




    #layer_range = [0,1]
   # for l in layer_range :
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


    print('Prediction : %s  layer : %s Average activation weights of Ben : %s  Adv : %s  GT : %s'%
    (i,1,sum_b/len(begning_sample_act)
    ,sum_a/len(adv_sample_act),
    sum_g/len(ground_truth_act)))

    return


def expF(i):
    # This experiment tres to find nodes that are always active 
    targer_prediction = i

    adversarial_sample = Accessor('./adversarial/mnist/PGD/mnist_2')
    begning_sample = Accessor('./begnign/mnist/mnist_2')
    ground_truth = Accessor('./Ground_truth/mnist/mnist_2')
    threshhold = 0
    print('attack PGD')

    begning_sample_act = begning_sample.get_label_by_prediction(targer_prediction)
    adv_sample_act = adversarial_sample.get_label_by_prediction(targer_prediction)
    ground_truth_act = ground_truth.get_label_by_prediction(targer_prediction)
    #Removing outliers
   
   
    counter = 0


    layer_range = [0,1]
    for l in layer_range:

        always_active_nodes_b = []
        for i in range(0,begning_sample_act[0].set_layer_range(l,l).get_number_of_node()):
            always_active_nodes_b.append(i)

        for i,x in enumerate(begning_sample_act) :
            x.set_layer_range(l,l)
            active_nodes = []
            for j,y in enumerate(x.flatten()):
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
        for i in range(0,adv_sample_act[0].set_layer_range(l,l).get_number_of_node()):
            always_active_nodes_adv.append(i)
        for i,x in enumerate(adv_sample_act) :
            x.set_layer_range(l,l)
            active_nodes = []
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
        for i in range(0,ground_truth_act[0].set_layer_range(l,l).get_number_of_node()):
            always_active_nodes_gt.append(i)
        for i,x in enumerate(ground_truth_act) :
            x.set_layer_range(l,l)
            active_nodes = []
            for j,y in enumerate(x.flatten()):
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
        print(f'Prediction {targer_prediction} layer : {l}')
        print('number of nodes always active benign : %s'%(len(always_active_nodes_b)))
        print(always_active_nodes_b)
        print('number of nodes always active Adv : %s'%(len(always_active_nodes_adv)))
        print(always_active_nodes_adv)
        print('number of nodes always active GT : %s'%(len(always_active_nodes_gt)))
        print(always_active_nodes_gt)
        
        
        print('Number of common nodes betwen Always_active_begnign and always_active_gt : %s'%(counter_b))
        print('Numver of common nodes betwen always_active_adv and always_active_gt : %s'% (counter_adv))


def expG(i):
    #Purpsoe fo this experiment is to compute the frquency of node sactivations

    targer_prediction = i
    adversarial_sample = Accessor('./adversarial/mnist/fgsm/mnist_1')
    begning_sample = Accessor('./begnign/mnist/mnist_1')
    ground_truth = Accessor('./Ground_truth/mnist/mnist_1')
    print('attack fgsm')
    
    begning_sample_act = begning_sample.get_label_by_prediction(targer_prediction)
    adv_sample_act = adversarial_sample.get_label_by_prediction(targer_prediction)
    ground_truth_act = ground_truth.get_label_by_prediction(targer_prediction)


    
    frequency_gt = []
    frequency_be = []
    frequency_adv = []

    threshhold = 0
    layer_range = [1,2]
    for l in layer_range :

        nb_nodes_in_model = begning_sample_act[0].set_layer_range(l,l).get_nb_nodes()

        #compute frequencies , each array will hold the percentage of activations of all nodes len(arra) = nb_nodes
        for i in range(0,nb_nodes_in_model):
            frequency_gt.append(0)
            frequency_be.append(0)
            frequency_adv.append(0)


    
        for i in ground_truth_act:
            i.set_layer_range(l,l)
            flat = i.flatten()

            #Remove activations with more nodes than expected
            #if(len(flat) !=nb_nodes_in_model):
            #    print('Inconsistant')
            #   continue

            for index,j in enumerate(flat):
                if( j>threshhold):
                    frequency_gt[index]+=1 / len(ground_truth_act)



        for i in begning_sample_act:
            i.set_layer_range(l,l)
            flat = i.flatten()

            #Remove activations with more nodes than expected
            #if(len(flat) !=nb_nodes_in_model):
            #    continue
            
            for index,j in enumerate(flat):
                if( j>threshhold):
                    frequency_be[index]+=1 /len(begning_sample_act)



        for i in adv_sample_act:
            i.set_layer_range(l,l)
            flat = i.flatten()

            #Remove activations with more nodes than expected
            #if(len(flat) !=nb_nodes_in_model):
            #    continue
        
            for index,j in enumerate(flat):
                if( j>threshhold):
                    frequency_adv[index]+=1 /len(adv_sample_act)


        print(frequency_gt)
      
        
        #compute distance between frequencies b-gt  adv-gt 
        begnig_gt = np.array(frequency_be) - np.array(frequency_gt)
        adv_gt = np.array(frequency_adv) -np.array(frequency_gt)

        print('distance between Benign and Gt: %s' %(np.average(np.absolute(begnig_gt))))
        print('distance between Adversarial and Gt: %s ' %(np.average(np.absolute(adv_gt))))


        #compute nodes that are alway active (frequency =1)
        always_active_b = []
        always_active_gt = []
        always_active_adv = []

        counter =0
        for x,y,z in zip(frequency_adv,frequency_be,frequency_gt):
            if(x==1):
                frequency_adv.append(counter )
            if(y == 1 ):
                always_active_b.append(counter)

            if(z==1) :
                always_active_gt.append(counter)
            counter+=1

        counter_b = 0
        # compute common nodes in b-gt adv-gt
        for i in always_active_b:
            if( i in always_active_gt):
                counter_b+=1
        counter_adv = 0 
        for i in always_active_adv:
            if(i in always_active_gt):
                counter_adv +=1
        print(f'Prediction {targer_prediction} layer : {l}')
        print('number of nodes always active benign : %s'%(len(always_active_b)))
        print(always_active_b)
        print('number of nodes always active Adv : %s'%(len(always_active_adv)))
        print(always_active_adv)
        print('number of nodes always active GT : %s'%(len(always_active_gt)))
        print(always_active_gt)
        
        
        print('Number of common nodes betwen Always_active_begnign and always_active_gt : %s'%(counter_b))
        print('Numver of common nodes betwen always_active_adv and always_active_gt : %s'% (counter_adv))

        print()
        
    '''
    fig, axs = plt.subplots(3,figsize=(20,5))
    fig.suptitle ('Activation frequency on prediction %s  mnist_1 fgsm'%(targer_prediction))
    
    axs[0].set(xlabel='Nodes', ylabel='Frequencies')
    axs[1].set(xlabel='Nodes', ylabel='Frequencies')

    # To plot nodes activations
    limit = 400
    offset = 0

    X_axis = np.arange(limit) 
    axs[0].hist( frequency_gt,label = 'GT',color ='grey')
    axs[1].hist( frequency_be,label = 'BE',color='green')
    axs[2].hist( frequency_adv, label = 'ADV',color="red")
 
   
  
    plt.legend()
    plt.show()
    '''  

def expH(i):
    #Dispersation index

    targer_prediction = i
    adversarial_sample = Accessor('./adversarial/cifar10/FGSM/cifar10_1')
    begning_sample = Accessor('./begnign/cifar10/cifar10_1')
    ground_truth = Accessor('./Ground_truth/cifar10/cifar10_1')

    begning_sample_act = begning_sample.get_label_by_prediction(targer_prediction,verbose=0)
    adv_sample_act = adversarial_sample.get_label_by_prediction(targer_prediction,verbose=0)
    ground_truth_act = ground_truth.get_label_by_prediction(targer_prediction,verbose=0)

    #layer_range = [1,2,3]        
    # Calcualte the dispersation index 
    #for l in layer_range:
    index_ben = 0
    for i in begning_sample_act:
        #i.set_layer_range(l,l)
        index_ben += i.dispersation_index() / len(begning_sample_act)
    
    index_adv = 0
    for i in adv_sample_act:
        #i.set_layer_range(l,l)
        index_adv += i.dispersation_index() / len(adv_sample_act)


    index_gt = 0
    for i in ground_truth_act:
        #i.set_layer_range(l,l)
        index_gt += i.dispersation_index() / len(ground_truth_act)

    #print('Prediction %s  layer  Dispersation index Gt : %s  Benign : %s  Adv : %s '%(index_gt,index_ben,index_adv))
    print(f'prediction {targer_prediction} layer : {0} dispersation index  GT : {index_gt} Benign : {index_ben} Adv : {index_adv}')

def plot(i): 
    targer_prediction = i
    adversarial_sample = Accessor('./adversarial/mnist/pgd/mnist_1')
    begning_sample = Accessor('./begnign/mnist/mnist_1')
    ground_truth = Accessor('./Ground_truth/mnist/mnist_1')
    print('attac pgd')
    begning_sample_act = begning_sample.get_label_by_prediction(targer_prediction)
    adv_sample_act = adversarial_sample.get_label_by_prediction(targer_prediction)
    ground_truth_act = ground_truth.get_label_by_prediction(targer_prediction)


    
    sample = 0



    '''
    fig, axs = plt.subplots(3,figsize=(20,5))
    fig.suptitle ('Activations Weight on prediction %s  Cuckoo '%(targer_prediction))
    
    axs[0].set(xlabel='Nodes', ylabel='Weights')
    axs[1].set(xlabel='Nodes', ylabel='Weights')

    # To plot nodes activations
    limit = 64
    offset = 0
    
    X_axis = np.arange(limit) 
    axs[0].hist(  ground_truth_act[sample].flatten(),label = 'GT',color ='grey')
    axs[1].hist(  begning_sample_act[sample].flatten(),label = 'BE',color='green')
    axs[2].hist(  adv_sample_act[sample].flatten() , label = 'ADV',color="red")
    '''

    print(len( begning_sample_act[sample].flatten()))
    print(len( adv_sample_act[sample].flatten()))
    print(len(ground_truth_act[sample].flatten()))
   
   
   
    #plot activations scatter
    begning_sample_act[sample].plot("green", "Sample act benign %s" %(i))
    adv_sample_act[sample].plot("red", "Sample act adv %s" %(i))
    ground_truth_act[sample].plot("grey", "Sample act GT %s" %(i))

    

    plt.show()


    return 



if __name__ == "__main__":
    for i in range(1,10):
        expD(i)





