from utils import *
from Activations import *
from Accessor import Accessor
import sys

#Generate activations for selected dataset under selected attacks




dataset = 'cifar10'
#In Uppercase
attack = 'PGD'
shape =784
model_name= "cifar10_1"
 



# todo 1 add cifar10
model = get_model(model_name)
(X_train, Y_train), (X_test, Y_test) = get_dataset(dataset,True,True,False)


def generate_train_activations():
    #optimize this
    #Takes 30 mn to finish
    counter = 0
    for i,x in enumerate(X_train):
        '''
        # remove wrongly predicted samples in train data (Activating this will increase time a loooooot)
        pred = np.argmax(model.predict(np.reshape(x,(-1,28,28)),verbose =0))
        if(pred != Y_train[i]):
            continue
        '''

        if(isinstance(Y_train, pd.DataFrame)) : y = Y_train.iloc[i]
        else : y = Y_train[i]
    
        generate_and_save_activations(model,x,i,y,"./Ground_Truth/"+dataset+"/"+model_name)
        printProgressBar(i + 1, X_train.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
        counter+=1
       
    print("Generated and saved  Train set activations dataset %s size " %(dataset))


def generate_test_activation_adv():
    index = 0    
    for i,x in enumerate(X_test):
      
        if(isinstance(Y_test, pd.DataFrame)) : y = Y_test.iloc[i]
        else : y = Y_test[i]
    
        x = np.reshape(x,(-1,32,32,3))
        x = generate_attack_tf(model,x,y,attack)
        generate_and_save_activations(model,x,index,y, "./adversarial/"+dataset+"/"+attack+"/"+model_name)
        index+=1
        printProgressBar(index + 1, X_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
    print("Generating Adversarial Activations Csv for Dataset %s  under attack %s " % (dataset,attack))


def generate_test_activations_begnign():
    index = 1
    true_count = 1
    for i,x in enumerate(X_test):
        
        if(isinstance(Y_test, pd.DataFrame)) : y = Y_test.iloc[i]
        else : y = Y_test[i]

        if(generate_and_save_activations(model,x,index,y,"./begnign/"+dataset+"/"+model_name)) : 
            true_count+=1
        if(index %100 ==0):
            print(f'accuracy so far : {true_count/index*100} %')

        index+=1
        printProgressBar(index + 1, X_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
    print("Generated Begnign Activations Csv for Dataset %s    " % (dataset))




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



        









