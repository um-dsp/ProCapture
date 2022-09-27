from utils import *
from Activations import *
from Accessor import Accessor


#Generate activations for selected dataset under selected attacks

attack = None
dataset = 'mnist'


offswitch = input("Generating Activations for dataset %s attack %s (Y,n) :" % (dataset,attack))
if(offswitch == "n"):
    exit()



model = get_model("mnist")
(X_train, Y_train), (X_test, Y_test) = get_dataset('mnist',True,False,True)



#Generates acivations for a given model and input and saves in corresponding folder
def generate_and_save_activations(self,model,input,index,label,attack,dataset):
    ac =  get_layers_activations(model,input)
    input = np.reshape(input,(-1,28,28))
    prediction =np.argmax(model.predict(input,verbose=0)[0])
    activations = [item for sublist in ac for item in sublist]
    a = Activations(index,label,prediction,activations,attack)
    a.save_csv(dataset)
    return a


index = 0
if (attack =='fgsm') :
    print("Generating Adversarial Activations Csv for Dataset %s  under attack %s " % (dataset,attack))
    for x,y in zip(X_test,Y_test):
        x = generate_fgsm(x,y,model)
        generate_and_save_activations(model,x,index,y,attack,dataset)
        index+=1
        printProgressBar(index + 1, X_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)

else:
    print("Generating Begnign Activations Csv for Dataset %s    " % (dataset))
    for x,y in zip(X_test,Y_test):
        generate_and_save_activations(model,x,index,y,None,dataset)
        index+=1
        printProgressBar(index + 1, X_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)










