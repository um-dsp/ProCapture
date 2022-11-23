import gnn.GNN as GNN
import gnn.gnn_utils as ut
from Accessor import Accessor
import numpy as np
from Net_Subgraph import Net 
import tensorflow._api.v2.compat.v1 as tf

def client_optimizer_fn(learning_rate,name):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate,name =name)

if __name__ == "__main__":
    tf.disable_v2_behavior() 

    #adversarial_sample = Accessor('./adversarial/mnist/fgsm/mnist_1')
    begning_sample = Accessor('./begnign/mnist/mnist_1')
    begning_sample_act = begning_sample.get_label_by_prediction(0)
    #adv_sample_act = adversarial_sample.get_all()
    expected_nb_nodes = 384

    sample = begning_sample_act[0]
    sample.set_layer_range(1,2)
    activations = sample.get_activations_set()
    print(len(sample.flatten()))
    edges = []
    nodes =  []

    # if fully connected :
    
    l = 0
    id = 0
    while(l<len(activations)-1):
        for i in activations[l]:
            nodes.append([i,id])
            for j in activations[l+1]:
                edges.append([i,j,id])
        for i in activations[l+1]:
            nodes.append([i,id])
        l+=1

    nodes = np.array(nodes,dtype='int')
    edges = np.array(edges,dtype='int')
    
    inp ,arcnode, graphnode = ut.from_EN_to_GNN(edges,nodes)
    labels = np.array([0] * len(inp))



    input_dim =inp.shape[0]
    output_dim =labels.shape[0]
    state_dim = 1
    max_it = 50
    num_epoch = 10000
    threshold = 0.01
    learning_rate = 0.01
     
    net = Net(inp.shape[0],state_dim=1,output_dim = labels.shape[0])
    g = GNN.GNN(net, input_dim, output_dim, state_dim,  max_it, client_optimizer_fn, learning_rate, threshold, graph_based=False,
                tensorboard=False)
    loss, it = g.Train(inp, arcnode, labels, j)

