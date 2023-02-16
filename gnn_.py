import torch
from torch_geometric.data import Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader
from Accessor import Accessor
from utils import *
from captum.attr import Saliency, IntegratedGradients


begning_sample = Accessor('./begnign/cifar10/cifar10_1')
adversarial_sample = Accessor('./adversarial/cifar10/pgd/cifar10_1')


adv_sample_act = adversarial_sample.get_all(limit=11)
begning_sample_act = begning_sample.get_all(limit=11)



def to_dataset(activations,state):

    dataset = []
    index=  0
    for act in activations:
        index+=1
        printProgressBar(index + 1, len(activations), prefix = 'Progress:', suffix = 'Complete', length = 50)
        nodes = []
        for i in act.activations_set:
            for j in i:
                nodes.append([j])
        
        source = []
        target = []
        start_index= 0
        for i in range(len(act.activations_set)-1):
            for s in range(len(act.activations_set[i])):
                for t in range(len(act.activations_set[i+1])):
                    if(act.activations_set[i+1][t] < 0.01 and act.activations_set[i+1][t]>-0.01):
                        continue 
                    source.append(s+start_index)
                    target.append(t+start_index+len(act.activations_set[i]))
            start_index= len(act.activations_set[i])
        x = torch.FloatTensor(nodes)
        y = torch.LongTensor([state])
        edge_index = torch.tensor([source,
                            target], dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index)
        dataset.append(data)
    return dataset

ben_dataset = to_dataset(begning_sample_act,0)
adv_dataset = to_dataset(adv_sample_act,1)  

print()
print(len(ben_dataset))
print(len(adv_dataset))



a = ben_dataset[0:9]
b = adv_dataset[0:9]
c =ben_dataset[9:10]
d =adv_dataset[9:10]
train_dataset = [*a, *b]
test_dataset = [*c,*d]
print(f'train dataset : {len(train_dataset)}')
print(f'test dataset : {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

'''
#represents node sfeatures => Nodes activations
x = torch.FloatTensor([[0],[0], [0], [0]])
y = torch.LongTensor([0])
edge_index = torch.tensor([[0, 2, 0, 3],
                           [1, 1, 3, 2]], dtype=torch.long)

data = Data(x=x, y=y, edge_index=edge_index)
dataset = [data,data,data,data,data,data,data,data]

'''



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels,2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        #x = x.relu()
        #x = self.conv3(x, edge_index)
       
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x



model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.



for epoch in range(1, 5):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

