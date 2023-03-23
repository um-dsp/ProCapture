from library.Accessor import Accessor
from library.attributionUtils import adversarial_detection_set
import torch
from library.train import train_adversrial_detection_model
from library.attributionUtils import NeuralNetMnist_1
import sys

supported_dataset = ['cifar10' ,'mnist', 'cuckoo','ember'] 
supported_attacks = ['FGSM','CW','PGD',"CKO",'EMBER',None]
pre_trained_models = ['cifar10_1','cuckoo_1','Ember_2','mnist_1','mnist_2','mnist_3']
folder = ['groundTruth' , 'begnign' ,'adversarial']

def parse_args():
    args= sys.argv
    if (len(args) !=6 ): raise ValueError('Wrong Arguments')
    dataset = args[1]
    if( dataset not in supported_dataset):
        raise ValueError(f'ProvMl only Supports {supported_dataset}')
        
    model_name = args[2]
    if(model_name not in pre_trained_models ):
        raise ValueError(f'ProvMl only Supports {pre_trained_models}')
        
    attack = args[3]
    if(attack not in supported_attacks):
          raise ValueError(f'ProvMl only Supports {supported_attacks}')   

    expected_nb_nodes = int(args[4])

    model_path = args[5]
    
  
    return dataset,attack,model_name,expected_nb_nodes,model_path



if __name__ == "__main__":
    dataset,attack,model_name,expected_nb_nodes,model_path = parse_args()
    print(f'\n[GRAPH LEARN] dataset: {dataset} | attack: {attack} | model: {model_name} \n')

    begning_accessor = Accessor('./begnign/'+dataset+'/' +model_name +'/')
    adversarial_accessor= Accessor('./Adversarial/'+dataset+'/'+attack +'/' +model_name +'/' )
    ground_truth_accessor = Accessor('./Ground_Truth/'+dataset+'/' +model_name +'/')

    begning_sample_act = begning_accessor.get_all()
    adv_sample_act = adversarial_accessor.get_all()
    gt_sample_act = ground_truth_accessor.get_all()


    # Transforms the activations to the folowing data set : x[activationA,activaitonB,...]  y= [1, 0 ,1...]
    X_adv,Y_adv=adversarial_detection_set(adv_sample_act,label = torch.tensor(1),expected_nb_nodes=expected_nb_nodes)
    X_ben,Y_ben=adversarial_detection_set(begning_sample_act,label = torch.tensor(0),expected_nb_nodes=expected_nb_nodes)
    X_gt ,Y_gt =adversarial_detection_set(gt_sample_act,label = torch.tensor(0),expected_nb_nodes=expected_nb_nodes)

    X = X_adv + X_ben + X_gt
    Y= Y_adv + Y_ben + Y_gt

    print(f'[GRAPH LEARN]  Training Model : X {len(X)} |  Y {len(Y)}')
    
    train_adversrial_detection_model(X,Y,model_path)
