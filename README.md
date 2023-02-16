# ProvML

# ProvML

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
