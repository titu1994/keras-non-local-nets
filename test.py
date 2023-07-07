'''A simple non-local block test'''
import numpy as np

from non_local import NonLocalBlock

def tensors(rank=3,L=3,range_nums=(1,5),dims=False):
    """
    Generate a random tensor and apply a non-local block operation.

    Parameters
    ------------
    rank : int
        Rank of the tensor (number of dimensions).
    L : int 
        characteristic side length of tensor
    range_nums: tuple 
        Range of random integers to generate the tensor values.
    dims : tuple or False)
        Custom dimensions for the tensor. If False, the tensor will have shape rank*[L].
    """
    tensor_shape = rank*[L] if not dims else dims
    x = np.random.randint(*range_nums,(tensor_shape))*1.0
    print (x)

    non_local_block = NonLocalBlock(intermediate_dim=None, compression=2, mode='embedded', add_residual=True)
    output = non_local_block(x)
    print(output)


for i in [3, 4,5]:
    tensors(i)

tensors(dims=(100,99,102))   #stress case
