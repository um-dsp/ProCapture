
This is for Pip library ignore for now 
ProvMl
=======

This Library allows you to extract and save neural networks inner layer activations. This library contributes
 to the ideology of white-box neural networks and implements the trivial tools for neural networks inner functionaliteis
 Analysis, It furthermroe implements a series of functionalities that could be utilized for analysis such as metrics,explainability
 and Statistics.

Activation Generation
----------------------

    ..  code-block:: python
    from library.generate_activations import generate_activations <br />
    (X_train, y_train), (X_test, y_test) = get_dataset('mnist',False,True) <br />
    model = get_model('mnist_1') <br />
    generate_activations(X_adv,y_test,model,'./test') <br />

Activations Parsing 
--------------------
    ..  code:: python
    begning_sample = Accessor('./begnign/test/') <br />
    begning_sample_act = begning_sample.get_all(limit=1000) <br />

Run Metrics 
--------------
    ..  code:: python
    from library.metrics import expD
    begning_sample_act = begning_sample.get_all(limit=1000)   
    expD(begning_sample_act,adv_sample_act,ground_truth_act)


Docs 
======
Class Generate Activations
------------------------

..  code:: python
    generate_activations(X,Y,model , path) 

`X`  
    : (np.array) input set

'Y' 
    : (np.array) corresponding labels

`model`  
    : (tansorflow model) model

`file_path`  
    : (string) path to folder where to sae activations

Class Accessor 
--------------

..  code:: python
    get_all(collapse='avg',sub_ration = 0,limit = float('+inf'),start=0) 

collapse 
    : for CNN activations each node activation is an aray: this parameter selects strategy to transform the array to a representative float
sub_ration 
    : when set to n accesor will return n % of the total activations
limit 
    : set a limit to the number of activations parsed
start   
    : when set to n the parser will stat parsing from file n in folder




    