#### Activation File Format :

The activations generation will save to the folder a set of CSV | TXT files ,each file represents the activations of a individual input throught the set model and using the attack if mentioned :

- **File Name** : {label}-{prediction}\_{id in dataset} .txt |.csv
- **File Content** : .txt files are used for Convolutional Neural network wher each node acitvaiton is represented by an ndarray
  .csv files are used fro feedforward network where the ouput of The node is a float
- `Validator.py`: offers a st of function to run verifications on the activations folder. such as compute_accuracy
- `Accessor.py` : offers a set of functionalities that imports and parses the activations files into an array of activations Object. This Class also offer many parameters to allow acces using label , prediction or get all activations ..
