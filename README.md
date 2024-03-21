# ProvML: Inference Provenance-Driven Characterization of Machine Learning in Benign and Adversarial Settings

We describe the supported datasets, attacks and pre-trained models provided with our code. <br />

### Datasets, Pre-trained Models, and Attacks:


- **Datasets**: MNIST is autimatically loaded via Keras. To test ProvML on malware data, you need to download the [CuckooTraces]([link here](https://drive.google.com/file/d/11GgjGVEXQAAz09J_T7sziJdS6vF14cwu/view?usp=sharing)) and [EMBER] ([link here](https://ember.elastic.co/ember_dataset_2018_2.tar.bz2)) datasets and add them in the folder `./data/`. By default ProvML will look for them in that path<br />

- **Pre-trained Models**: We offer pre-trained models: mnist_1 , mnist_2, mnist_3, cuckoo_1, and ember_1 <br />
  These models are availabe to download [here](https://drive.google.com/drive/folders/1a0kdq4waz8SXU9gThsUmKsR0YTSuaEWO?usp=share_link). Once downloaded to 'ProvML/models/' directory, the 'model.txt' file has the model architecture details of each model.
  
- **Attacks**: ProvML supports the following attacks : <br />
  MNIST: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD) Auto PGD  with DLR loss function (APGD-DLR) , Square <br />
  CuckooTraces: Attack progressively flips up to first n 0 bits to 1 until it evades the model (we name this attack 'CKO') <br />
  EMBER => This attack progressively perturbs features within valid value ranges/options until the model changes its prediction from malware to benign (we call this attack `EMB') <br />

***

### Downloading ProvML and Installing Dependencies:
```$ git clone https://github.com/um-dsp/DeepProv.git ```

```$ cd DeepProv ```

```$ pip install -r requirements.txt ```
***

### Inference Activation Graph Extraction:

- ` activations_extractor.py`: The first step for our characterization approach is to extract activations of the target model `model`. It takes the following parameters in the given order:

- **Dataset Name**:  mnist | cuckoo |ember <br />
- **Pre-Trained Model Name**: cuckoo_1 | ember_1 | mnist_1 | mnist_2 | mnist_3 <br />
- **Folder**: Ground_Truth | Benign | Adversarial <br>
   It is required to use these exact folder names. This parameter sets what folder will the generated activations be saved, the default file path is
  `folder/dataset_name/model_name/<attack>/`. 
  (**Note**: Make Sure to create the folder with the above path before running the activation generation)
- **Attack Name**: FGSM | APGD-DLR | PGD |square|CKO |EMBER | None <br />
 - **tasks**: default is to get emperical characterization and graph for the extrating graphs from the model inputs.<br /> 
  This parameter is optional.  if specified, ProvML will apply the attack on the dataset.
  (**Note**: if attack is None and the folder input is set to adversarial it will throw an Error. -stop parameter is to precise the number of batchs of graph to generate (1000 graphs per batch) <br />
  **Sample Commands :** <br />
    >   `python activations_extractor.py -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth -model_type pytorch -task default` <br />
     >   ` python activations_extractor.py -dataset mnist -model_name mnist_1 -folder Benign_pth -model_type pytorch -task default` <br />
     >   ` python activations_extractor.py -dataset mnist -model_name mnist_1 -folder Benign_pth -model_type pytorch -task default -attack FGSM` <br />
 ***         
The model activations of ground truth, test benign and adversarial data are stored in each respective folder in text files named as [true label]\_[predicted label]\_[index] (e.g., 0_0_150.txt). Each file contains the values of every node in every layer of the model for that specific sample.   

This commands is specifically needed to train GNN later on. <br /> <br />
    >  `python activations_extractor.py -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth -model_type pytorch -task graph -stop 10` <br />
    >  `python activations_extractor.py -dataset mnist -model_name mnist_1 -folder Ground_Truth_pth -model_type pytorch -task graph -attack FGSM -stop 10` <br />
To train GNN and save it using the generated graphs use the following command :  <br /> <br />
   >  `python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -attack FGSM  -epochs 5 -save True` <br />


To explain the GNN and visualize the structred attributions of the graphs use the following commands : <br /> <br />
  >  `python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_mnist_2_FGSM_pytorch -attack FGSM  -expla_mode Saliency -attr_folder /data/attributions_data/` <br />
    >  `python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_mnist_2_FGSM_pytorch -expla_mode Saliency -attr_folder /data/attributions_data/ ` <br />

***          

### Empirical Characterization:
 
 Use [Empirical_Characterization.ipynb](/Empirical_Characterization.ipynb) to compute the proposed graph-related metrics for empirical characterization.

---
### Structured Characterization

 Use [Structured_Characterization.ipynb](/Structured_Characterization.ipynb) to compute the proposed graph-related metrics for empirical characterization.

#### Graph Feature learning model: training a model `graph_model` on the extracted graph data

` train_on_graph.py`: We train another model called `graph_model` that learns the NN graph (activations) of the target model `model`. This model should be also stored should be initiated and stored in a .pt file <br />

This step utilizes the activations extracted in the previous step, To Train and test a predefined model on the activations set use the CLI command with the following parameters: Dataset Name, Model Name and Attack
  (The Above arguments will just be used to locate the needed activations in the project folder).
  The command also expects the following arguments: <br />
- Model Path : represents the path to save the feature extraction model that is trained to seperate activations of benign and adversarial samples <br/>
  (**Note**: We have pre-trained models on Ground_Truth benign and FGSM data, and tested on Benign and FGSM test data.

  **Sample Commands:** <br />
  To train a graph model and save it on given dataset:
  > `python train_on_graph.py -dataset mnist  -model_name mnist_1 -attack FGSM  -model_path ./models/mnist_1.pt -task default` <br />
    To train a GNN model and save it for a given dataset:
  > `python train_on_graph.py -dataset mnist  -model_name mnist_1 -folder Ground_Truth_pth -model_type pytorch -task graph -attack PGD  -epochs 50 -save True` <br />
    To generate the explanation attributes for a given GNN model
     > `python train_on_graph.py -dataset mnist  -model_name mnist_1 -folder Benign_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_mnist_1_FGSM_pytorch -expla_mode IntegratedGradients` <br /> 
  
#### Attribution: Perform ML explanation on the `graph_model` to identify relevant nodes [Still not ready]

- ` gen_attributions.py`: this file explains how to transform generated activations to dataset and train an torch adversarial detection model. ` attributionUtils.py` holds different predefined architecture that cover all the dataset we research and produce satisfactory performance.
  ` Attributions :` in the same file we showcase the steps to generate the attributions of the models on a batch of input,
  
 --- 
  
  - ` Attribution.py`: The thrid step: is to perform **Attributions** on the trained `graph_model`. This file should include a fucntion that takes as input a `graph_model` and `data_name`, the model should be imported automaticcaly from `Models/[data_name]/[[graph_model_name]`. Example: `Models/MNIST/graph_MNIST.pth`

