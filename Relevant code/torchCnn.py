import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Accessor import *
import numpy as np 
from sklearn.model_selection import train_test_split
import itertools
from attributionUtils import NeuralNetEmber, NeuralNetCuckoo_1, NeuralNetCifar_10,attribtuions_to_polarity,NeuralNetMnist_1,NeuralNetMnist_2,NeuralNetMnist_3,get_attributes,activations_to_adv_detector_set
import random
from attributionUtils import KNN 


begning_sample = Accessor('./begnign/mnist/mnist_1')
adversarial_sample = Accessor('./adversarial/mnist/FGSM/mnist_1')
#alternate_test = Accessor('./adversarial/cifar10/fgsm/cifar10_1')
expected_nb_nodes = 420
begning_sample_act = begning_sample.get_all()
adv_sample_act = adversarial_sample.get_all()
#alternate_act =  alternate_test.get_all()


'''
    To categorical  :
    [1. 0.] => 0   =>Benign

'''


X,Y =activations_to_adv_detector_set(adv_sample_act,begning_sample_act,expected_nb_nodes)


'''
model = NeuralNetEmber()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
accuracy = 0
#while(accuracy <80):
X_train, X_test,y_train, y_test = train_test_split(X,Y ,random_state=104, test_size=0.25, shuffle=True)
print(f'X_test len {len(X_test)}')
print(f'Y_test len {len(y_test)}')

    #Train Model
for epoch in range(5):
    #Load in the data in batches using the train_loader object
    correct =0
    for x, y in  zip(X_train,y_train):  

        x = x[None, :]
        # Forward pass
        outputs = model(x)
        outputs = torch.squeeze(outputs)

        #print(torch.argmax(outputs),torch.argmax(y))

        loss = criterion(outputs, y)
        correct += 1 if (torch.argmax(outputs) == torch.argmax(y)) else 0 
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 10, loss.item()))
    print(f'accuracy {correct/len(X_train) *100}')
accuracy = correct/len(X_train) *100

#torch.save(model, './advDetectionModels/torch_ember.pt')
#exit()
'''
    
model = torch.load('./advDetectionModels/torch_Mnist_1_FGSM.pt')

criterion = nn.CrossEntropyLoss()

'''
def compute_accuracy_torch():
    correct= 0
    for x, y in  zip(X_train,y_train):  
    
            x = x[None, :]
            # Forward pass
            outputs = model(x)
            outputs = torch.squeeze(outputs)

            #print(torch.argmax(outputs),torch.argmax(y))

            loss = criterion(outputs, y)
            correct += 1 if (torch.argmax(outputs) == torch.argmax(y)) else 0 
            # Backward and optimize
            print(f'accuracy {correct/len(X_train) *100}')
    accuracy = correct/len(X_train) *100

'''


def get_nodes_impact_for_one_label(label):
    positive, negative = attribtuions_to_polarity(X_test,y_test,model,label)


    nodes_freq_p = [0 for i in range(expected_nb_nodes)]
    nodes_freq_n = [0 for i in range(expected_nb_nodes)]

    print(f'number of sampels studies {len(positive)}')
    for i in positive : 
        for j in i : 
            nodes_freq_p[j]+=1 / len(positive)

    for i in negative : 
        for j in i : 
            nodes_freq_n[j]+=1 / len(negative)


    always_pos = []
    always_neg = []
    both = []
    neutral = []
    for index,(i,j) in enumerate(zip(nodes_freq_p,nodes_freq_n)):
        if(i == 0 and j !=0 ) :
            always_pos.append(index)
        if(i != 0 and j ==0 ) :
            always_neg.append(index)
        if(i == 0 and j == 0 ) :
            neutral.append(index)
        if(i != 0 and j !=0 ) :
            both.append(index)

    print(f'For label {label}')
    print(f'Number of  always positive nodes : {len(always_pos)}')
    print(f'Number of  always negative nodes : {len(always_neg)}')
    print(f'Number of  neutral nodes : {len(neutral)}')
    print(f'Number of  undisicive nodes : {len(both)}')

    return always_pos,always_neg,neutral,both

nodes_weight_per_label = [[[],[]] for i in range(expected_nb_nodes)]

def get_nodes_weight_per_label(label):
    negative_nodes = [[] for i in range(expected_nb_nodes)]
    positive_nodes = [[] for i in range(expected_nb_nodes)]

    for index,input in enumerate(X):
        if(Y[index][1] != label) : continue
        attributes = get_attributes(input,model,label)
        for i,x in enumerate(attributes) :
            if(x>0): positive_nodes[i].append(input[i])
            if(x<0): negative_nodes[i].append(input[i])
    return positive_nodes,negative_nodes

def get_avg_number_of_nodes_per_state(label):
    pos = 0
    neg = 0
    counter = 0
    for index,input in enumerate(X):
        if(Y[index][1] != label) : continue
        attributes = get_attributes(input,model,label)
        for i,x in enumerate(attributes):
            if(x>0) : pos+=1
            if(x<0) : neg+=1
        counter+=1
    print(f'positive {pos} negative {neg}')
    return pos/counter, neg/counter

    
def plot_diffrences(ben,ben_,adv,adv_):
    p = []
    for i in range(len(ben)):
        all_ben = ben[i] + ben_[i]
        all_adv = adv[i] + adv_[i]
        if(not all_ben or not all_adv): continue
     
        avg_ben = np.average(all_ben)
        avg_adv = np.average(all_adv)
        print(f'Average ben : {np.average(avg_ben)}  adv {np.average(avg_adv)}')
        p.append(avg_adv-avg_ben)
        
    plt.xlabel("Nodes")
    plt.ylabel("Difference")
    y = np.arange(len(x))
    plt.title("avg_weight(ben) - avg_weight(adv) Mnist_1 FGSM")
    plt.bar(y,x)
    plt.show()

def box_plot(ben,ben_,adv,adv_):
    #banign / dav in benign samples
    ben,adv = get_nodes_weight_per_label(1)
    #banign / dav in adversarail samples
    adv_,ben_ = get_nodes_weight_per_label(0)
    case0 = 0
    case1 = 0 
    case2 = 0
    case3 = 0
    nb = 0 
    x = []
    while(nb <10):
        i = random.randint(1,expected_nb_nodes)
        all_ben = ben[i] + ben_[i]
        all_adv = adv[i] + adv_[i]
        if(not all_ben or not all_adv): continue
        print(f'Average ben : {np.average(all_ben)}  adv {np.average(all_adv)}')
        print(f'Range [ {min(all_ben)} :{max(all_ben)}]   adv {min(all_adv)} :{max(all_adv)} ')
        avg_ben = np.average(all_ben)
        avg_adv = np.average(all_adv)
        min_ben =min(all_ben)
        max_ben= max(all_ben)
        min_adv =min(all_adv)
        max_adv= max(all_adv)
        #x[ 'B' +str(i)] = all_ben
        #x['A'+str(i)] = all_adv
        nb+=1
        x.append(avg_adv - avg_ben)    
        if(max_ben <min_adv):
            case1 +=1
        elif(max_adv<min_ben):
            case2+=1
        elif(avg_ben<avg_adv):
            case3+=1
        elif(avg_ben>avg_adv):
            case4+=1

    fig, ax = plt.subplots()
    box = ax.boxplot(x.values(),patch_artist=True)
    for i,v in enumerate(box["boxes"]):
        if(i%2 == 0 ): continue
        plt.setp(v, color="red")

        print(i)
    ax.set_xticklabels(x.keys())

    plt.show()


label = 0
clusteringX = []
clusteringY = []
for index,input in enumerate(X):
    attributes = get_attributes(input,model,label)
    mul = np.multiply(attributes,input)
    clusteringX.append(np.array(mul))
    clusteringY.append(int(Y[index][1].item()))

X_train, X_test,y_train, y_test = train_test_split(clusteringX,clusteringY ,random_state=104, test_size=0.25, shuffle=True)

'''
a = torch.Tensor([
    [1, 1],
    [0.88, 0.90],
    [-1, -1],
    [-1, -0.88]
])

b = torch.LongTensor([3, 3, 5, 5])

c = torch.Tensor([
    [-0.5, -0.5],
    [0.88, 0.88]
])

knn = KNN(a, b)
print(knn(c))
'''
X_test = X_test[1:100]
y_test = y_test[1:100]

X_train = torch.FloatTensor(np.array(X_train))
y_train = torch.LongTensor(np.array(y_train))
X_test = torch.FloatTensor(np.array(X_test))

knn = KNN(X_train, y_train)
pred = knn(X_test)

match =0
for x, y in zip(pred,y_test):
    if(x==y):
        match+=1
    print(f'Accuracy : {match/len(X_test) * 100}')
