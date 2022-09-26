from utils import *
from Activations import *
from Loader import Loader


#Generate activations for selected dataset under selected attacks

attack= input('Generate Activations begnign or under attack (b,a)')
if(attack =="a"):
    attack = "fgsm"
else :
    attack = None

dataset = 'mnist'
l = Loader('./adversarial',"./begnign","mnist")


model = get_model("mnist")
(X_train, Y_train), (X_test, Y_test) = get_dataset('mnist',True,False,True)



index = 0
if (attack =='fgsm') :
    print("Generating Activations Csv for Dataset %s  under attack %s " % (dataset,attack))
    for x,y in zip(X_test,Y_test):
        x = generate_fgsm(x,y,model)
        l.generate_and_save_activations(model,x,index,y,attack)
        index+=1
        printProgressBar(index + 1, X_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)

else:
    print("Generating Activations Csv for Dataset %s  under attack %s " % (dataset,attack))
    for x,y in zip(X_test,Y_test):
        l.generate_and_save_activations(model,x,index,y,None)
        index+=1
        printProgressBar(index + 1, X_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)










