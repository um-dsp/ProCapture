from tkinter import N
from xml.etree.ElementInclude import LimitedRecursiveIncludeError
from utils import get_model
from utils import get_dataset
from utils import printProgressBar
import numpy as np
from Accessor import Accessor


model = get_model("mnist")

(X_train, Y_train), (X_test, Y_test) = get_dataset('mnist',True,False,True)

def expA ():
    #In this expirament we evaluate average of hamilton index of adversarial test data relative to train data
    #(Ground truth) and comapre it to the same result of begnign test data
    # Purpose is to evaluate if the begnign activations are 'closer' to train activations then adversarial activations
    adversarial_sample = Accessor('./adversarial/mnist/fgsm')
    begning_sample = Accessor('./begnign/mnist')
    ground_truth = Accessor('./Ground_truth/mnist')

    class_activations_adv = adversarial_sample.get_by_label('0')
    class_activations_begnign = begning_sample.get_by_label('0')
    ground_truth_activations = ground_truth.get_by_label('0')

    indexes = []
    limit = 1000
    for y in class_activations_adv :
        for i,x in enumerate(ground_truth_activations) :
            if i > limit :
                break
            aux = []
            aux.append(y.hamilton_index(x,0,4))
        indexes.append(np.average(aux))
    print('For Adv samples, Hamilton distance avg', np.average(indexes))

    indexes = []

    for y in class_activations_begnign :
        for i,x in enumerate(ground_truth_activations) :
            if i > limit :
                break
            aux = []
            aux.append(y.hamilton_index(x,0,4))
        indexes.append(np.average(aux))
    print('For begnign samples, Hamilton distance avg', np.average(indexes))



if __name__ == "__main__":
  

   
    







