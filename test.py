from Activations import Activations
from Accessor import Accessor
A = Accessor('./begnign/mnist')
activation_ = A.get_instance_by_index(3,None)
activation1 = A.get_instance_by_index(4,None)

print(activation_.hamilton_index(activation1,1,3))