# ProvML: Inference Provenance-Driven Characterization of Machine Learning in Adversarial Settings

TODO:
Please complete the following files/folders using relevant code in `Library` folder:

1- We need to complete the following three python files that represents the skeleton of the tool. They have to be implemented in a unified way and independantly from the used model or datasets. Thus, the taget model and datasets should be set as parameters by the user.

- ` activation_extractor.py`: The first step on our characterization approach is to extract activations of the target model `model`. This model should be pretrained and previously stored at `Models/[data_name]/`
- ` learn-graph.py`: The Second step: We train another model called `graph_model` that learns the NN graph (activations) of the target model `model`. This model should be also stored at `Models/[data_name]/`.
- ` Attribution.py`: The thrid step: is to perform **Attributions** on the trained `graph_model`. This file should include a fucntion that takes as input a `graph_model` and `data_name`, the model should be imported automaticcaly from `Models/[data_name]/[[graph_model_name]`. Example: `Models/MNIST/graph_MNIST.pth`

2- Our extensive experiments, including plots, compultations and test that we performed for emperical and structured characterization has be included in a jupyter notebook file `characterization.ipynb`. This file should list code chunks of all our experiments ans plots with description of each chunk of code.

3- [Optional] A jupyter notebook that runs our code as library for in-depth functionalities

---

#DOCS

- ` gen_activations.py`: The first step on our characterization approach is to extract activations of the target model `model`. Import data and your pretrained model and utilzie the functions in library/generate_activations.py to generate and save the activations

### Dataset , Pretrained Models , Attacks :

ProvMl provides a set of dataset ,pretrained models and attacks in its implementation , Using the CLI command will limit the user to these models. Use Library if you need to extend to other dataset,attacks. <br />
`Datasets` : Cifar10 and Mnist. Download cuckoo and Ember dataset and put them in the folder `./data/`. By default ProvMl will look for them in that path<br />
`Pretrained Models` : we offer pretrained models: mnist_1 , mnist_2 , mnist_3 , cifar10_1 ,cuckoo_1 and ember_1 <br />
` Attacks`: ProvMl supports the following attacks : <br />

- Cifar10, Mnist => FGSM, PGD
- Cuckoo => Reverse first n bits attack (CKO)
- Ember => Developed personalized attack (EMB)

### Activation Generation Process

To Use activaiton generation file use the CLI with the following parameters :

- `Dataset Name` : cifar10 | mnist | cuckoo |ember <br />
- `Pre-Trained Model Name` : cifar10_1 |cuckoo_1 | Ember_2 | mnist_1 | mnist_2 | mnist_3 <br />
- `Folder` : groundTruth | begnign | adversarial <br />
  this parameter sets what folder will the generated activations be saved, the default file path is
  `folder/dataset_name/model_name/<attack>/`
- `Attack Name` : FGSM | CW | PGD |CKO |EMBER | None <br />
  this parameter is optional , if mentioned, ProvMl will apply the attack on the dataset.
  (**note** : if attack is None and the folder input is set to adversarial it will throw an Error) <br />
  Sample Commands : <br />
  > `py gen_activations.py mnist mnist_1  adversarial FGSM` <br /> `py gen_activations.py mnist mnist_1 groundTruth  ` <br />

###### Activation File Format :

The activations generation will save to the folder a set of CSV | TXT files ,each file represents the activations of a individual input throught the set model and using the attack if mentioned :

- **File Name** : {label}-{prediction}\_{id in dataset} .txt |.csv
- **File Content** : .txt files are used for Convolutional Neural network wher each node acitvaiton is represented by an ndarray
  .csv files are used fro feedforward network where the ouput of The node is a float
- `Validator.py`: offers a st of function to run verifications on the activations folder. such as compute_accuracy
- `Accessor.py` : offers a set of functionalities that imports and parses the activations files into an array of activations Object. This Class also offer many parameters to allow acces using label , prediction or get all activations ..

# Metrics :

- `run_metrics.py` : Once you have the generated activations of adversarial, benign and Groundtruth, instanciate an Accessor classwith the path of each. Then leverage the experiment expD() -> expI() to compute the different metrics
  Metrics : average number of active ndoes , activaions weight, nodes frequencies, always active nodes, Dispersation index, Entropy index.
- ` gen_attributions.py`: this file explains how to transform generated activations to dataset and train an torch adversarial detection model. ` attributionUtils.py` holds different predefined architecture that cover all the dataset we research and produce satisfactory performance.
  ` Attributions :` in the same file we showcase the steps to generate the attributions of the models on a batch of input,

This software enables users to extract and store the hidden layer activations of a neural network that they select. Additionally, it offers a range of metrics to evaluate the differences between datasets, such as the average number of active nodes and frequency distances. Moreover, we offer a machine learning-based approach to compute the truth state of a prediction based on the inner logits of the neural network. Lastly, we provide a methodology for explainability techniques to gain further insights into the activations.

Activations.py:
This class represents the hidden layer activations for a specific dataset or attack, (a graph). It includes various functionalities that can be applied to the activations_set, such as computing the number of active nodes and dispersion index.

Accessor.py:
This class serves as a crawler that allows for loading the activations from txt/csv files and parsing them to the Activations class. It implements various functionalities, such as getting elements by prediction, by label, or retrieving all elements.

generate_activations.py:
This function encapsulates the necessary functionalities to pass an input through a model, extract the hidden layer weights, and save them in a user-defined file. The user can import these functions to generate custom activations or trigger them through the terminal to create default activation samples.

metrics.py:
This file contains the code for each metric discussed in the paper. ExpA to ExpD , Each experiment requires three file paths that represent the adversarial, benign, and ground truth data. This function should only be used once the required activations have been extracted into a text file.
