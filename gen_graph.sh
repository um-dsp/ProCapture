#!/bin/bash
#python activations_extractor.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -stop 10

#python activations_extractor.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -stop 10 -attack FGSM

#python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -attack FGSM  -epochs 5 -save True
python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_mnist_2_FGSM_pytorch -attack FGSM  -expla_mode Saliency -attr_folder /data/attributions_data/
python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_mnist_2_FGSM_pytorch -expla_mode Saliency -attr_folder /data/attributions_data/              



#python activations_extractor.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -stop 10 -attack APGD-DLR

#python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_   mnist_2_APGD-DLR_pytorch -attack APGD-DLR  -expla_mode Saliency
#python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_   mnist_2_APGD-DLR_pytorch -expla_mode Saliency
#python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -attack APGD-DLR  -epochs 5 -save True

#python activations_extractor.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -stop 10
#python activations_extractor.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -stop 10 -attack APGD-DLR
#python activations_extractor.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -stop 10 -attack FGSM
#python activations_extractor.py -dataset    mnist  -model_name    mnist_2 -folder Benign_pth -model_type pytorch -task graph 
#python activations_extractor.py -dataset    mnist  -model_name    mnist_2 -folder Adversarial_pth -model_type pytorch -task graph -attack APGD-DLR
#python activations_extractor.py -dataset    mnist  -model_name    mnist_2 -folder Adversarial_pth -model_type pytorch -task graph  -attack FGSM
#python train_on_graph.py -dataset cuckoo  -model_name cuckoo_1 -folder Ground_Truth_pth -model_type pytorch -task graph -attack CKO  -epochs 5 -save False
#python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task graph -attack FGSM  -epochs 5 -save True
#python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_   mnist_2_FGSM_pytorch -attack FGSM  -expla_mode Saliency
#python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_   mnist_2_FGSM_pytorch -expla_mode Saliency
#python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_   mnist_2_APGD-DLR_pytorch -attack APGD-DLR  -expla_mode Saliency
#python train_on_graph.py -dataset    mnist  -model_name    mnist_2 -folder Ground_Truth_pth -model_type pytorch -task GNN_explainer -model_path models/GNN_   mnist_2_APGD-DLR_pytorch -expla_mode Saliency
