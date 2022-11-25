import gnn.GNN as GNN
import gnn.gnn_utils as ut
from Accessor import Accessor
import numpy as np
from Net_Subgraph import Net 
import tensorflow._api.v2.compat.v1 as tf
import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn import runner

'''
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
'''

def model_fn(gtspec: tfgnn.GraphTensorSpec):
  """Builds a simple GNN with `ConvGNNBuilder`."""
  convolver = tfgnn.keras.ConvGNNBuilder(
      lambda edge_set_name, receiver_tag: tfgnn.keras.layers.SimpleConv(
          lambda: tf.keras.layers.Dense(32, activation="relu"),
          "sum",
          receiver_tag=receiver_tag,
          sender_edge_feature=tfgnn.HIDDEN_STATE),
      lambda node_set_name: tfgnn.keras.layers.NextStateFromConcat(
          lambda: tf.keras.layers.Dense(32, activation="relu")),
      receiver_tag=tfgnn.SOURCE)
  return tf.keras.Sequential([
      convolver.Convolve() for _ in range(4)  # Message pass 4 times.
  ])


# in neeed arr = [[i]]  [activation_weights]
# in need srouce arra =[i(flattened)] target = arr[i(flattened)]
graph = tfgnn.GraphTensor.from_pieces(
   node_sets={
       "cnn_node": tfgnn.NodeSet.from_fields(
           sizes=tf.constant([3]),
           features={
               "index": tf.ragged.constant(
                   [[0],
                    [1],
                    [2]]),
               "activation": tf.constant([1, 0, 1]),
           })},

   edge_sets={
       "cnn_edge": tfgnn.EdgeSet.from_fields(
           sizes=tf.constant([3]),
           adjacency=tfgnn.Adjacency.from_indices(
               source=("paper", tf.constant([1, 2, 2])),
               target=("paper", tf.constant([0, 0, 1]))))
     
               })
'''
with tf.io.TFRecordWriter('./gnn_data.txt') as writer:
  for _ in range(1000):
    example = tfgnn.write_example(graph)
    writer.write(example.SerializeToString())
'''

graph_schema = tfgnn.read_schema("./schema.pbtxt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)


trainer = runner.KerasTrainer(
    strategy=tf.distribute.experimental.CentralStorageStrategy(),
    model_dir="...",
     steps_per_epoch=1,  # global_batch_size == 128
    validation_per_epoch=2,

    #validation_steps=1)  
)

task = runner.RootNodeBinaryClassification(node_set_name="nodes")

train_ds_provider = runner.TFRecordDatasetProvider(file_pattern="./gnn_data.txt")
print(train_ds_provider)
runner.run(train_ds_provider=train_ds_provider,valid_ds_provider=train_ds_provider
,model_fn =model_fn,optimizer_fn=tf.keras.optimizers.Adam,trainer=trainer,task=task,global_batch_size=128,gtspec=gtspec)