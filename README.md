# ProvML: Inference Provenance-Driven Characterization of Machine Learning in Benign and Adversarial Settings

We describe the supported datasets, attacks and pre-trained models provided with our code. <br />

### Datasets, Pre-trained Models, and Attacks:

- **Datasets**: MNIST and CIFAR10 are autimatically loaded via Keras. To test ProvML on malware data, you need to download the [CuckooTraces](link here) and [EMBER] (link here) datasets and add them in the folder `./data/`. By default ProvML will look for them in that path<br />

- **Pre-trained Models**: We offer pre-trained models: mnist_1 , mnist_2, mnist_3, cifar10_1, cuckoo_1, and ember_1 <br />
  These models are availabe to download [here](https://drive.google.com/drive/folders/1a0kdq4waz8SXU9gThsUmKsR0YTSuaEWO?usp=share_link). Once downloaded to 'ProvML/models/' directory, the 'model.txt' file has the model architecture details of each model.
  
- **Attacks**: ProvML supports the following attacks : <br />
  MNIST, CIFAR10: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD) <br />
  CuckooTraces: Attack progressively flips up to first n 0 bits to 1 until it evades the model (we name this attack 'CKO') <br />
  EMBER => This attack progressively perturbs features within valid value ranges/options until the model changes its prediction from malware to benign (we call this attack `EMB') <br />

***

### Downloading ProvML and Installing Dependencies:
```$ git clone https://github.com/um-dsp/ProvML.git ```

```$ cd ProvML ```

```$ pip install -r requirements.txt ```
***

### Inference Activation Graph Extraction:

- ` activations_extractor.py`: The first step for our characterization approach is to extract activations of the target model `model`. It takes the following parameters in the given order:

- **Dataset Name**: cifar10 | mnist | cuckoo |ember <br />
- **Pre-Trained Model Name**: cifar10_1 |cuckoo_1 | ember_1 | mnist_1 | mnist_2 | mnist_3 <br />
- **Folder**: Ground_Truth | Benign | Adversarial <br>
   It is required to use these exact folder names. This parameter sets what folder will the generated activations be saved, the default file path is
  `folder/dataset_name/model_name/<attack>/`. 
  (**Note**: Make Sure to create the folder with the above path before running the activation generation)
- **Attack Name**: FGSM | CW | PGD |CKO |EMBER_att | None <br />
  This parameter is optional.  if specified, ProvML will apply the attack on the dataset.
  (**Note**: if attack is None and the folder input is set to adversarial it will throw an Error) <br />
  **Sample Commands :** <br />
    >  `python activations_extractor.py mnist mnist_1 Ground_Truth  ` <br />
    >  `python activations_extractor.py mnist mnist_1 Benign  ` <br />
    > `python activations_extractor.py mnist mnist_1 Adversarial FGSM` <br />
    > `python activations_extractor.py cuckoo cuckoo_1 Adversarial CKO` <br />
    > `python activations_extractor.py ember ember_1 Adversarial EMB` <br />
   
 To extract Adversarial activations on training data (Ground_Truth), we use: <br />
    >  `python activations_extractor.py mnist mnist_1 Ground_Truth FGSM ` <br />
This command is specifically needed to train graph_model later on. <br /> <br />
The model activations of ground truth, test benign and adversarial data are stored in each respective folder in text files named as [true label]\_[predicted label]\_[index] (e.g., 0_0_150.txt). Each file contains the values of every node in every layer of the model for that specific sample. 
***

### Empirical Characterization:
 
 Use [Empirical_Characterization.ipynb](/Empirical_Characterization.ipynb) to compute the proposed graph-related metrics for empirical characterization.

---
### Structured Characterization

#### Graph Feature learning model: training a model `graph_model` on the extracted graph data

` train_on_graph.py`: We train another model called `graph_model` that learns the NN graph (activations) of the target model `model`. This model should be also stored should be initiated and stored in a .pt file <br />

This step utilizes the activations extracted in the previous step, To Train and test a predefined model on the activations set use the CLI command with the following parameters: Dataset Name, Model Name and Attack
  (The Above arguments will just be used to locate the needed activations in the project folder).
  The command also expects the following arguments: <br />
- Model Path : represents the path to save the feature extraction model that is trained to seperate activations of benign and adversarial samples <br/>
  (**Note**: We have pre-trained models on Ground_Truth benign and FGSM data, and tested on Benign and FGSM test data.

  **Sample Commands:** <br />

  > `python train_on_graph.py cifar10 cifar10_1 FGSM ./models/cifar10_1.pt` <br />
  > `python train_on_graph.py mnist mnist_1 FGSM ./models/mnist_1.pt` <br />


#### Attribution: Perform ML explanation on the `graph_model` to identify relevant nodes [Still not ready]

- ` gen_attributions.py`: this file explains how to transform generated activations to dataset and train an torch adversarial detection model. ` attributionUtils.py` holds different predefined architecture that cover all the dataset we research and produce satisfactory performance.
  ` Attributions :` in the same file we showcase the steps to generate the attributions of the models on a batch of input,
  
 --- 
  
  - ` Attribution.py`: The thrid step: is to perform **Attributions** on the trained `graph_model`. This file should include a fucntion that takes as input a `graph_model` and `data_name`, the model should be imported automaticcaly from `Models/[data_name]/[[graph_model_name]`. Example: `Models/MNIST/graph_MNIST.pth`

