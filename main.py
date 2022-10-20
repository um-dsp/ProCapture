from nntplib import GroupInfo
from utils import generate_attack_tf, get_model
from utils import get_dataset
from utils import printProgressBar
import numpy as np
from Accessor import Accessor
import matplotlib.pyplot as plt

model = get_model("mnist")

(X_train, Y_train), (X_test, Y_test) = get_dataset('mnist',True,False,True)

def expA ():
    #In this expirament we evaluate average of hamming index of adversarial test data relative to train data
    #(Ground truth) and comapre it to the same result of begnign test data
    # Purpose is to evaluate if the begnign activations are 'closer' to train activations then adversarial activations
   
    user_inp = 0
   
    adversarial_sample = Accessor('./adversarial/mnist/PGD')
    begning_sample = Accessor('./begnign/mnist')
    ground_truth = Accessor('./Ground_truth/mnist')

    class_activations_adv = adversarial_sample.get_by_label(user_inp)
    class_activations_begnign = begning_sample.get_by_label(user_inp)
    ground_truth_activations = ground_truth.get_by_label(user_inp)
    
    limit = 7000
    
    indexes = [[]]
    for j,y in enumerate(class_activations_adv) :
        for i,x in enumerate(ground_truth_activations) :
            if i >limit:
                break
            indexes[j].append(y.hamilton_index(x,0,3))
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
            indexes[j].append(y.hamilton_index(x,0,3))
        printProgressBar(j, len(class_activations_begnign), prefix = 'Progress:', suffix = 'Complete', length = 50)
        if(j<len(class_activations_begnign)-1):
                    indexes.append([])        
    arr = np.array(indexes)
    np.savetxt('./hamilton_indexes/begnign.csv',arr,fmt='%s')       



def expB(inp):
    # Purpose of this experiment is to compute the number of nodes active(output>0) in each activation
    #Compute median and average of each label
    #Comapred between metrics for adversarial,begnign, and training

    #Get activation nodes stats
    user_inp = inp
   
    adversarial_sample = Accessor('./adversarial/mnist/fgsm')
    begning_sample = Accessor('./begnign/mnist')
    ground_truth = Accessor('./Ground_truth/mnist')

    class_activations_adv = adversarial_sample.get_by_label(user_inp)
    class_activations_begnign = begning_sample.get_by_label(user_inp)
    ground_truth_activations = ground_truth.get_by_label(user_inp)

    activated_counter=0
    begnign_holder =[]
    for i,x in enumerate(class_activations_begnign):
        nb = x.compute_nb_active_nodes()
        begnign_holder.append(nb)
    nb=0
    adversarial_holder = []
    for i,x in enumerate(class_activations_adv):
        nb = x.compute_nb_active_nodes()
        adversarial_holder.append(nb)

    nb=0
    ground_truth_holder = []
    for i,x in enumerate(ground_truth_activations):
        nb = x.compute_nb_active_nodes()
        ground_truth_holder.append(nb)
    

    print('LABEL : %s AVG : BEGN: %s ADV : %s GT: %s'%(inp,
        np.average(np.array(begnign_holder)),
        np.average(np.array(adversarial_holder)),
        np.average(np.array(ground_truth_holder))
    ))    
    print('LABEL :%s MEDIAN : BEGN: %s ADV : %s GT: %s'%(inp,
        np.median(np.array(begnign_holder)),
        np.median(np.array(adversarial_holder)),
        np.median(np.array(ground_truth_holder))
    ))  

def expC(dataset ="mnist",attack="FGSM"):
    # Purpsoe of experiemnt to compute the average disturbance of input image
    # (determinig if on average imafe label pixels were increased or decreaser)
    # attempt to explain why adv input pishes more nodes to activate
    model = get_model(dataset)
    (X_train, Y_train), (X_test, Y_test) = get_dataset(dataset,True,False,True)
    aux = []
    for i,x in enumerate(X_test) :
        x = x.reshape(-1,28,28)
        x_adv = generate_attack_tf(model,x,Y_test[i],attack)
        perturbation_avg = np.average(x_adv-x)
        aux.append(perturbation_avg)
    print(aux)
    print('ON average Input images pixel values were distorted by %s' %(np.average(np.array(aux))))
    #MNIST FGSM  +0.001339769
    #MNIST PGD   +0.00017820533


def expD(i):
    targer_prediction = i
    adversarial_sample = Accessor('./adversarial/mnist/PGD')
    begning_sample = Accessor('./begnign/mnist')
    ground_truth = Accessor('./Ground_truth/mnist')

    begning_sample_act = begning_sample.get_label_by_prediction(targer_prediction)
    adv_sample_act = adversarial_sample.get_label_by_prediction(targer_prediction)
    ground_truth_act = ground_truth.get_label_by_prediction(targer_prediction)

    miss = []
    for x in adv_sample_act:
        x.set_layer_range(1,3)
        if not x.get_truth_value():
            miss.append(x)
    avg_active_node_miss = 0
    for y in miss :
        avg_active_node_miss += y.compute_nb_active_nodes() / len(miss)



    hit = []
    for x in begning_sample_act:
        x.set_layer_range(1,3)
        if  x.get_truth_value():
            hit.append(x)
    avg_active_node_hits = 0
    for y in hit :
        avg_active_node_hits += y.compute_nb_active_nodes() / len(hit)



    gt_nb = 0
    for y in ground_truth_act :
        y.set_layer_range(1,3)
        gt_nb += y.compute_nb_active_nodes() / len(ground_truth_act)

    print('Avg of active nodes for prediction : %s Gt : %s   Begnign : %s  Adversarial : %s' %(targer_prediction,gt_nb,avg_active_node_hits,avg_active_node_miss))



def expE(i):
    #purpose of this experiment is to detemine the average weight of activations 
    # and comapre adversarial,begnign and training
    targer_prediction = i
    adversarial_sample = Accessor('./adversarial/mnist/fgsm')
    begning_sample = Accessor('./begnign/mnist')
    ground_truth = Accessor('./Ground_truth/mnist')

    begning_sample_act = begning_sample.get_label_by_prediction(targer_prediction)
    adv_sample_act = adversarial_sample.get_label_by_prediction(targer_prediction)
    ground_truth_act = ground_truth.get_label_by_prediction(targer_prediction)

    sum_b = 0
    for x in begning_sample_act:
        x.set_layer_range(1,3)
        sum_b += x.get_average_weight() 


    sum_a = 0
    for x in adv_sample_act:
        x.set_layer_range(1,3)
        sum_a += x.get_average_weight()


        
    sum_g = 0
    for x in ground_truth_act:
        x.set_layer_range(1,3)
        sum_g += x.get_average_weight()
    print('Label : %s' , (i) )
    print(' Average activation weights of Ben : %s  Adv : %s  GT : %s'%
    (sum_b/len(begning_sample_act)
    ,sum_a/len(adv_sample_act),
    sum_g/len(ground_truth_act)))

    return

def expF(i):
    # This experiment tres to find nodes that are always active 
    targer_prediction = i

    adversarial_sample = Accessor('./adversarial/mnist/PGD')
    begning_sample = Accessor('./begnign/mnist')
    ground_truth = Accessor('./Ground_truth/mnist')

    begning_sample_act = begning_sample.get_label_by_prediction(targer_prediction)
    adv_sample_act = adversarial_sample.get_label_by_prediction(targer_prediction)
    ground_truth_act = ground_truth.get_label_by_prediction(targer_prediction)
    

    nb_nodes_in_model = len(adv_sample_act[0].set_layer_range(1,2).flatten())

    print('Number of nodes per model :%s' %(nb_nodes_in_model))
    always_active_nodes_b = []
    for i in range(0,nb_nodes_in_model):
        always_active_nodes_b.append(i)
    for i,x in enumerate(begning_sample_act) :
        x.set_layer_range(1,2)
        active_nodes = []
        for j,y in enumerate(x.flatten()):
            if(y>0):
                active_nodes.append(j)
        pop_counter = 0
        for index,g in enumerate(always_active_nodes_b):
            if(g not in active_nodes):
                always_active_nodes_b.pop(index)
                pop_counter += 1
        if(len(always_active_nodes_b) == 0 ):
            break



    always_active_nodes_adv = []
    for i in range(0,nb_nodes_in_model):
        always_active_nodes_adv.append(i)
    for i,x in enumerate(adv_sample_act) :
        x.set_layer_range(1,2)
        active_nodes = []
        for j,y in enumerate(x.flatten()):
            if(y>0):
                active_nodes.append(j)
        pop_counter = 0
        for index,g in enumerate(always_active_nodes_adv):
            if(g not in active_nodes):
                always_active_nodes_adv.pop(index)
                pop_counter += 1
        if(len(always_active_nodes_adv) == 0 ):
            break
   

    always_active_nodes_gt = []
    for i in range(0,nb_nodes_in_model):
        always_active_nodes_gt.append(i)
    for i,x in enumerate(ground_truth_act) :
        x.set_layer_range(1,2)
        active_nodes = []
        for j,y in enumerate(x.flatten()):
            if(y>0):
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
    
    print('Number of common nodes betwen Always_active_begnign and always_active_gt : %s'%(counter_b))
    print('Numver of common nodes betwen always_active_adv and always_active_gt : %s'% (counter_adv))


def expG(i):
    #Purpsoe fo this eperiment is to compute the frquency of node sactivations

    targer_prediction = i
    adversarial_sample = Accessor('./adversarial/mnist/fgsm')
    begning_sample = Accessor('./begnign/mnist')
    ground_truth = Accessor('./Ground_truth/mnist')

    begning_sample_act = begning_sample.get_label_by_prediction(targer_prediction)
    adv_sample_act = adversarial_sample.get_label_by_prediction(targer_prediction)
    ground_truth_act = ground_truth.get_label_by_prediction(targer_prediction)


    frequency_gt = []
    frequency_be = []
    frequency_adv = []
    for i in range(0,400):
        frequency_gt.append(0)
        frequency_be.append(0)
        frequency_adv.append(0)

    
    for i in ground_truth_act:
        i.set_layer_range(1,2)
        for index,j in enumerate(i.flatten()):
            if( j>0):
                frequency_gt[index]+=1 / len(ground_truth_act)

    for i in begning_sample_act:
        i.set_layer_range(1,2)
        for index,j in enumerate(i.flatten()):
            if( j>0):
                frequency_be[index]+=1 /len(begning_sample_act)

    for i in adv_sample_act:
            i.set_layer_range(1,2)
            for index,j in enumerate(i.flatten()):
                if( j>0):
                    frequency_adv[index]+=1 /len(adv_sample_act)

    begnig_gt = np.array(frequency_be) - np.array(frequency_gt)
    adv_gt = np.array(frequency_adv) -np.array(frequency_gt)

    print('distance between Benign and Gt: %s' %(np.average(np.absolute(begnig_gt))))
    print('distance between Adversarial and Gt: %s ' %(np.average(np.absolute(adv_gt))))
    



    '''
    limit = 50
    offset = 0
    X_axis = np.arange(limit)
    plt.figure(figsize=(20, 3))
    plt.bar(X_axis - 0.2, frequency_gt[offset:limit], 0.2, label = 'GT',color ='grey')
    plt.bar(X_axis , frequency_be[offset:limit], 0.2, label = 'BE',color='green')
    plt.bar(X_axis+0.2 , frequency_adv[offset:limit], 0.2, label = 'ADV',color="red")
    plt.xlabel('Node Id')
    plt.ylabel('Activation frequency on prediction %s'%(targer_prediction))
    plt.legend()
    plt.show()
    
    '''
        



    

    


    return 



if __name__ == "__main__":
 
    expF(7)







