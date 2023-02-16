from utils import *
from Activations import *
from Accessor import Accessor
import sys

#Generate activations for selected dataset under selected attacks




dataset = 'ember'
#In Uppercase
attack = 'EMBER'
shape =784
model_name= "ember_2"
 



(X_train, Y_train), (X_test, Y_test) = get_dataset(dataset,normalize=False,categorical = True)
model = load_model('./models/' +model_name+'.h5')


def select_only_one_label (X,Y,label):
    aux = []
    for m,n in enumerate(Y):
        if(np.argmax(n)== label):
            aux.append(X[m])
    return np.array(aux)

def is_wrong_prediction (model,x,y,i):
    pred = np.argmax(model.predict(x,verbose =0))
    return pred == y


def generate_activations(X,Y,model,file_path):

    counter = 1
    correct_predictions = 0
    for i,x in enumerate(X):
        # Catgorized label to label
        if(isinstance(Y, pd.DataFrame)) : y = Y.iloc[i]
        else : y = Y[i]


        #Reshape needed for K backendl ogits extractions
        x = np.expand_dims(x,axis= 0)
        #generate and save activations nd return if sucessfull prediction or not
        if(generate_and_save_activations(model,x,i,y,file_path)):
            correct_predictions+=1

        #Print the accuracy so far to monitor hidden layer extraction
        if(counter %100 ==0):
            print(f'accuracy so far : {correct_predictions/counter*100} %')

        printProgressBar(i + 1, X_train.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
        counter+=1
    print("Generated and saved set activations dataset to %s " %(file_path))

def generate_train_activations():
    #optimize this
    #Takes 30 mn to finish
       

    counter = 0
    correct_predictions = 0
    for i,x in enumerate(X_train):
      
        # Catgorized label to label
        if(isinstance(Y_train, pd.DataFrame)) : y = Y_train.iloc[i]
        else : y = Y_train[i]

        x =  np.reshape(x,(-1,x.shape[0]))

        #generate and save activations nd return if sucessfull prediction or not
        if(generate_and_save_activations(model,x,i,y,"./Ground_Truth/"+dataset+"/"+model_name)):
            correct_predictions+=1

        #Print the accuracy so far to monitor hidden layer extraction
        if(counter %100 ==0):
            print(f'accuracy so far : {correct_predictions/counter*100} % ', end='\r')

        printProgressBar(i + 1, X_train.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
        counter+=1

    print(" \n Generated and saved  Train set activations dataset %s size " %(dataset),end='\r')


def generate_test_activation_adv():

    count = 1   
    true_count = 1


    for i,x in enumerate(X_test):
       
        if(isinstance(Y_test, pd.DataFrame)) : y = Y_test.iloc[i]
        else : y = Y_test[i]

        x = generate_attack_tf(model,x,y,attack)
        
        x =  np.reshape(x,(-1,x.shape[0]))

        if(generate_and_save_activations(model,x,i,y, "./adversarial/"+dataset+"/"+attack+"/"+model_name)):
            true_count+=1
        if(count %100 ==0):
            print(f'accuracy so far : {true_count/count*100} %')
        count+=1

        printProgressBar(count + 1, X_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
    print("Generating Adversarial Activations Csv for Dataset %s  under attack %s " % (dataset,attack))


def generate_test_activations_begnign():

    correct_prediction = 1
    count = 0

    for i,x in enumerate(X_test):
        
        
        if(isinstance(Y_test, pd.DataFrame)) : y = Y_test.iloc[i]
        else : y = Y_test[i]

        x =  np.reshape(x,(-1,x.shape[0]))

        if(generate_and_save_activations(model,x,i,y,"./begnign/"+dataset+"/"+model_name)) : 
            correct_prediction+=1
        if(count %100 ==0):
            print(f'accuracy so far : {correct_prediction/count*100} %')

        count+=1
        printProgressBar(count + 1, X_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
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




        









