from tkinter import N
from Activations import Activations
from utils import get_model
from utils import get_dataset
from Loader import Loader


model = get_model("mnist")

(X_train, Y_train), (X_test, Y_test) = get_dataset('mnist',True,False,True)

l = Loader('./adversarial','./begnign',"mnist")
a = l.get_instance_by_label_prediction(0,0,None)
print(a.activations_set)