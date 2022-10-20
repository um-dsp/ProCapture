from utils import *
from Activations import *
from Accessor import Accessor
import sys

#Generate activations for selected dataset under selected attacks




dataset = 'mnist'
attack = 'fgsm'
shape =784
model_name= "mnist_2"




# todo 1 add cifar10
model = get_model(dataset)
(X_train, Y_train), (X_test, Y_test) = get_dataset(dataset,True,False,True)



def generate_train_activations():
    #optimize this
    #Takes 30 mn to finish
    counter = 0
    for i,x in enumerate(X_train):
        # remove wrongly predicted samples in train data
        pred = np.argmax(model.predict(np.reshape(x,(-1,28,28)),verbose =0))
        if(pred != Y_train[i]):
            continue
        generate_and_save_activations(model,x,i,Y_train[i],"./Ground_Truth/"+dataset+"/"+model_name)
        printProgressBar(i + 1, X_train.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
        counter+=1
    print("Generated and saved  Train set activations dataset %s size " %(dataset))


def generate_test_activation_adv():
    index = 0
    print("Generating Adversarial Activations Csv for Dataset %s  under attack %s " % (dataset,attack))
    for x,y in zip(X_test,Y_test):
        x = generate_attack_tf(model,x.reshape(-1,28,28),y,attack)
        generate_and_save_activations(model,x,index,y, "./adversarial/"+dataset+"/"+attack+"/"+model_name)
        index+=1
        printProgressBar(index + 1, X_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)

def generate_test_activations_begnign():
    index = 0
    print("Generating Begnign Activations Csv for Dataset %s    " % (dataset))
    for x,y in zip(X_test,Y_test):
        generate_and_save_activations(model,x,index,y,"./begnign/"+dataset+"/"+model_name)
        index+=1
        printProgressBar(index + 1, X_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)




if __name__ == "__main__":

    '''
      Run For full seeding         py .\generate_activations_run_time.py all 

    Sample commands
            py .\generate_activations_run_time.py train 
            py .\generate_activations_run_time.py test begnign 
            py .\generate_activations_run_time.py test adversarial  

    '''

    if(sys.argv[1] == 'train'):
        generate_train_activations()
    if(sys.argv[1] =='test'):
        if(sys.argv[2]== 'begnign'):
            generate_test_activations_begnign()
        if(sys.argv[2] == 'adversarial'):
            generate_test_activation_adv()
    if (sys.argv[1] =='all'):
            generate_train_activations()
            generate_test_activations_begnign()
            generate_test_activation_adv()



        









