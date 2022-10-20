from Activations import Activations
from Accessor import Accessor
from utils import * 
from numpy import *





if __name__ == '__main__':
  print(tf.test.is_built_with_cuda())
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
