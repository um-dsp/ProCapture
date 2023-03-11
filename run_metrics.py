from library.metrics import expD, expE,expF,expG,expH,expI
from library.Accessor import Accessor




if __name__ == "__main__":

    ############################ONLY USE THIS AFTER ACTIVATIONS ARE GENERATED ##############################
    ############################SAMPLE ACTIVATIONS OF MNIST PGD WERE SEEDE IN THE FOLLOWING FOLDERS ##############################


    #Instantiate and Accessor class with a folder containing activations csv/txt
    adversarial_sample = Accessor('./adversarial/pgd/')
    begning_sample = Accessor('./begnign/mnist_1/')
    ground_truth = Accessor('./Ground_truth/mnist_1/')

        
    #Accessor will load and parse the activations from csv/txt to an Activation Class
    #Accessor allows you to retreive the acivations by label, prediction, or get_all
    #Take a look at accessor for further details

    begning_sample_act = begning_sample.get_all(limit=1000)   
    adv_sample_act = adversarial_sample.get_all(limit=1000)
    ground_truth_act = ground_truth.get_all(limit=1000)


    #metrics.py containes a range of metrics that could be used ExpD -> ExpI
    #all are run using the same params
    expD(begning_sample_act,adv_sample_act,ground_truth_act)


    #You can also Experiment per prediction : 
    
    prediction = 0
    begning_sample_act = begning_sample.get_label_by_prediction(target_prediction= prediction,collapse='avg',limit=1000)   
    adv_sample_act = adversarial_sample.get_label_by_prediction(target_prediction = prediction,collapse='avg',limit=1000)
    ground_truth_act = ground_truth.get_label_by_prediction(target_prediction = prediction,collapse='avg',limit=1000)
    expI(begning_sample_act,adv_sample_act,ground_truth_act,prediction)