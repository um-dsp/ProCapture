from library.Accessor import Accessor
from library.attributionUtils import adversarial_detection_set
import torch
from library.train import train_on_activations
import sys
import numpy as np

supported_dataset = ['cifar10' ,'mnist', 'cuckoo','ember'] 
supported_attacks = ['FGSM','CW','PGD',"CKO",'EMBER',None]
pre_trained_models = ['cifar10_1','cuckoo_1','Ember_2','mnist_1','mnist_2','mnist_3']
folder = ['groundTruth' , 'benign' ,'adversarial']

def parse_args():
    args= sys.argv
    if (len(args) !=5 ): raise ValueError('Wrong Arguments')
    dataset = args[1]
    if( dataset not in supported_dataset):
        raise ValueError(f'ProvML only Supports {supported_dataset}')
        
    model_name = args[2]
    if(model_name not in pre_trained_models):
        raise ValueError(f'ProvML only Supports {pre_trained_models}')
        
    attack = args[3]
    if(attack not in supported_attacks):
        raise ValueError(f'ProvML only Supports {supported_attacks}')   

    #expected_nb_nodes = int(args[4])

    model_path = args[4]
    
  
    return dataset,attack,model_name,model_path



if __name__ == "__main__":
    dataset,attack,model_name,model_path = parse_args()
    print(f'\n[GRAPH LEARN] dataset: {dataset} | attack: {attack} | model: {model_name} \n')

    test_benign_accessor = Accessor('./Benign/'+dataset+'/' +model_name +'/')
    test_adv_accessor= Accessor('./Adversarial/'+dataset+'/'+attack +'/' +model_name +'/' )
    train_benign_accessor = Accessor('./Ground_Truth/'+dataset+'/' +model_name +'/')
    train_adv_accessor = Accessor('./Ground_Truth/'+dataset+'/FGSM/' +model_name +'/')
    
    print('Loading Benign training activations...')
    train_benign_act = train_benign_accessor.get_all()#limit=100)
    print('Loading Adversarial training activations...')
    train_adv_act = train_adv_accessor.get_all()#limit=100)
    print('Loading Benign testing activations...')
    test_benign_act = test_benign_accessor.get_all()#limit=10)
    print('Loading Adversarial testing activations...')
    test_adv_act = test_adv_accessor.get_all()#limit=10)
    #gt_sample_act = ground_truth_accessor.get_all()


    # Transforms the activations to the folowing data set : x[activationA,activaitonB,...]  y= [1, 0 ,1...]
    X_adv_train,Y_adv_train = adversarial_detection_set(train_adv_act,label = torch.tensor(1.0))
    X_ben_train,Y_ben_train = adversarial_detection_set(train_benign_act,label = torch.tensor(0.0))
    X_adv_test,Y_adv_test = adversarial_detection_set(test_adv_act,label = torch.tensor(1.0))
    X_ben_test,Y_ben_test = adversarial_detection_set(test_benign_act,label = torch.tensor(0.0))
    #X_gt ,Y_gt =adversarial_detection_set(gt_sample_act,label = torch.tensor(0),expected_nb_nodes=expected_nb_nodes)
    
    print('Shape of training adv activations:',X_adv_train.shape)
    print('Shape of training ben activations:',X_ben_train.shape)
    print('Shape of testing adv activations:',X_adv_test.shape)
    print('Shape of testing ben activations:',X_ben_test.shape)
    
    print('We sample equal number of adversarial and benign data ...')
    
    shape_min = np.min([X_adv_train.shape[0],X_ben_train.shape[0]])
    X_train = torch.cat((X_adv_train[:shape_min],X_ben_train[:shape_min]))# + X_gt
    Y_train= torch.cat((Y_adv_train[:shape_min], Y_ben_train[:shape_min]))# + Y_gt
    
    shape_min = np.min([X_adv_test.shape[0],X_ben_test.shape[0]])
    X_test = torch.cat((X_adv_test[:shape_min],X_ben_test[:shape_min]))# + X_gt
    Y_test= torch.cat((Y_adv_test[:shape_min], Y_ben_test[:shape_min]))# + Y_gt
    #print('Shape of all activations:',X.shape)
    print(f'[GRAPH LEARNING]  Training data : X_train {len(X_train)} |  Y_train {len(Y_train)}')
    print(f'[GRAPH LEARNING]  Testing data : X_test {len(X_test)} |  Y_test {len(Y_test)}')
    
    train_on_activations(X_train,Y_train,X_test,Y_test,model_name,model_path)
