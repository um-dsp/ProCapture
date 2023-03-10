# ProvML: Inference Provenance-Driven Characterization of Machine Learning in Adversarial Settings

TODO:
Please complete the following files/folders using relevant code in ```Relevant code``` folder:

1- We need to complete the following three python files that represents the  skeleton of the tool. They have to be implemented in a unified way and independantly from the used model or datasets. Thus, the taget model and datasets should be set as parameters by the user.

  * ``` activation_extractor.py```: The first step on our characterization approach is to extract activations of the target model ```model```. This model should be pretrained and previously stored at ```Models/[data_name]/```
  * ``` learn-graph.py```: The Second step: We train another model called ```graph_model``` that learns the NN graph (activations) of the target model ```model```. This model should be also stored at ```Models/[data_name]/```.
  * ``` Attribution.py```: The thrid step: is to perform **Attributions** on the trained ```graph_model```. This file should include a fucntion that takes as input a ```graph_model``` and ```data_name```, the model should be imported automaticcaly from ```Models/[data_name]/[[graph_model_name]```. Example: ```Models/MNIST/graph_MNIST.pth```

2- Our extensive experiments, including plots, compultations and test that we performed for emperical and structured characterization has be included in a jupyter notebook file ```characterization.ipynb```. This file should list code chunks of all our experiments ans plots with description of each chunk of code.

---
############
Documentation :

This software enables users to extract and store the hidden layer activations of a neural network that they select. Additionally, it offers a range of metrics to evaluate the differences between datasets, such as the average number of active nodes and frequency distances. Moreover, we offer a machine learning-based approach to compute the truth state of a prediction based on the inner logits of the neural network. Lastly, we provide a methodology for explainability techniques to gain further insights into the activations.

Activations.py:
This class represents the hidden layer activations for a specific dataset or attack, (a graph). It includes various functionalities that can be applied to the activations_set, such as computing the number of active nodes and dispersion index.

Accessor.py:
This class serves as a crawler that allows for loading the activations from txt/csv files and parsing them to the Activations class. It implements various functionalities, such as getting elements by prediction, by label, or retrieving all elements.

generate_activations.py:
This function encapsulates the necessary functionalities to pass an input through a model, extract the hidden layer weights, and save them in a user-defined file. The user can import these functions to generate custom activations or trigger them through the terminal to create default activation samples.

metrics.py:
This file contains the code for each metric discussed in the paper. ExpA to ExpD , Each experiment requires three file paths that represent the adversarial, benign, and ground truth data. This function should only be used once the required activations have been extracted into a text file.
