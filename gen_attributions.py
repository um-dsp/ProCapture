    
from library.Accessor import Accessor
from library.attributionUtils import get_attributes,adversarial_detection_set
from library.attributions import multiply_attributed_with_input,number_of_active_nodes
import numpy as np 
import torch
from library.train import train_adversrial_detection_model
from library.attributionUtils import NeuralNetMnist_1


if __name__ == "__main__":

 
    begning_accessor = Accessor('./Begnign/mnist/mnist_1')
    adversarial_accessor= Accessor('./Adversarial/mnist/FGSM/mnist_1')
    ground_truth_accessor = Accessor('./Ground_Truth/mnist/mnist_1')
    expected_nb_nodes = 420

    begning_sample_act = begning_accessor.get_all()
    adv_sample_act = adversarial_accessor.get_all()
    gt_sample_act = ground_truth_accessor.get_all()



    # Transforms the activations to the folowing data set : x[activationA,activaitonB,...]  y= [1, 0 ,1...]
    X_adv,Y_adv=adversarial_detection_set(adv_sample_act,label = torch.tensor(1),expected_nb_nodes=expected_nb_nodes)
    X_ben,Y_ben=adversarial_detection_set(begning_sample_act,label = torch.tensor(0),expected_nb_nodes =expected_nb_nodes)
    X_gt ,Y_gt =adversarial_detection_set(gt_sample_act,label = torch.tensor(0),expected_nb_nodes=expected_nb_nodes)

    X = X_adv + X_ben + X_gt
    Y= Y_adv + Y_ben + Y_gt
    train_adversrial_detection_model(X,Y,'./advDetectionModels/test.pt')

    model = torch.load('./advDetectionModels/test.pt')

    adv_attr =multiply_attributed_with_input(X_adv,Y_adv,model)
    ben_attr =multiply_attributed_with_input(X_ben,Y_ben,model)
    gt_attr =multiply_attributed_with_input(X_gt,Y_gt,model)

    avg_adv = [np.average(i) for i in adv_attr]
    ben_attr = [np.average(i) for i in ben_attr]
    gt_attr = [np.average(i) for i in gt_attr]

    print(f'Average Weight Adv :{np.average(avg_adv)} Ben : {np.average(ben_attr)} Gt : {np.average(gt_attr)} ')
