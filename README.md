# ProvML: Inference Provenance-Driven Characterization of Machine Learning in Adversarial Settings

We describe the supported Datasets, Attacks and the pre-trained models provided in our code. <br />

- **Datasets** : CIFAR10 and MNIST are autimatically loaded throught keras. To test ProvML on malware data, download Cuckoo-Traces and Ember datasets and add them in the folder `./data/`. By default ProvML will look for them in that path<br />
- **Pretrained Models** : we offer pretrained models: mnist_1 , mnist_2 , mnist_3 , cifar10_1 ,cuckoo_1 and ember_1 <br />
  Models are availabe to donwload [here](https://drive.google.com/drive/folders/1a0kdq4waz8SXU9gThsUmKsR0YTSuaEWO?usp=share_link)
  model.txt file has the metadata of each model
- **Attacks** : ProvML supports the following attacks : <br />
  Cifar10, Mnist => FGSM, PGD <br />
  Cuckoo => Reverse first null n bits attack (CKO) <br />
  Ember => Developed personalized attack (EMB) <br />

## Graph Extraction: Extraction Activations of a Neural Network model

- ` activations_extractor.py`: The first step for our characterization approach is to extract activations of the target model `model`. It takes the following parameters :

- **Dataset Name**: cifar10 | mnist | cuckoo |ember <br />
- **Pre-Trained Model Name** : cifar10_1 |cuckoo_1 | Ember_2 | mnist_1 | mnist_2 | mnist_3 <br />
- **Folder** : Ground_Truth | Benign | Adversarial <br />
  this parameter sets what folder will the generated activations be saved, the default file path is
  `folder/dataset_name/model_name/<attack>/`.
  (**note** : Make Sure to create the folder with the above path before running the generator)
- **Attack Name** : FGSM | CW | PGD |CKO |EMBER | None <br />
  this parameter is optional , if mentioned, ProvMl will apply the attack on the dataset.
  (**note** : if attack is None and the folder input is set to adversarial it will throw an Error) <br />
  **Sample Commands :** <br />
  > `python activations_extractor.py mnist mnist_1 Adversarial FGSM` <br /> `python activations_extractor.py mnist mnist_1 Ground_Truth  ` <br />
  
  The model activations of Ground truth, Test benign and adversarial data are stored in each respective folder. Check readme there. 


## Empirical Characterization:
 
 Use `Empirical_Characterization.ipynb` to compute the proposed graph-related metrics for empirical characterization.

---
## Structured Characterization [THIS PART IS STILL NOT READY]

### Graph Feature learning model: training a model `graph_model` on the extracted graph data

` train_on_graph.py`: The Second step: We train another model called `graph_model` that learns the NN graph (activations) of the target model `model`. This model should be also stored should be initiated and stored in a .pt file <br />

This step utilizes the activations extracted in the previous step, To Train a predefined model on the activations set use the CLI command with the following parameters: Dataset Name, Model Name and Attack
  (The Above arguments will just be used to locate the needed activations in the project folder).
  The command also expects the following arguments: <br />
- Expected Number Of Nodes : this number represents the number of nodes to expect in the activations, this is a safe guard against activation extraction errors and will ignore the sampels that have more nodes that expected
- Model Path : represents the path for the predefined .pt model <br/>
  (**note** : after training the model will be saved in the same file)
  The model will be trained across all samples in Ground_Truth , Adversarial and Begnign for a specific model/attack and for 30 epochs

  **Sample Commands :** <br />

  > `py learn-graph.py mnist mnist_1 FGSM 420 ./ModelsFolder/mnist_1.pt` <br /> `py gen_activations.py cuckoo cuckoo_1 62 ./ModelsFolder/cifar10_1.pt  ` <br />


### Attribution: Perform ML explanation on the `graph_model` to identify relevant nodes

- ` gen_attributions.py`: this file explains how to transform generated activations to dataset and train an torch adversarial detection model. ` attributionUtils.py` holds different predefined architecture that cover all the dataset we research and produce satisfactory performance.
  ` Attributions :` in the same file we showcase the steps to generate the attributions of the models on a batch of input,
  
 --- 
  
  - ` Attribution.py`: The thrid step: is to perform **Attributions** on the trained `graph_model`. This file should include a fucntion that takes as input a `graph_model` and `data_name`, the model should be imported automaticcaly from `Models/[data_name]/[[graph_model_name]`. Example: `Models/MNIST/graph_MNIST.pth`

