'''A simple non-local block test'''
import numpy as np

from non_local import NonLocalBlock

def test_tensors(rank=3):

    x = np.random.randint(1,5,(rank*[3]))*1.0
    print (x)

    non_local_block = NonLocalBlock(intermediate_dim=None, compression=2, mode='embedded', add_residual=True)
    output = non_local_block(x)
    print(output)

for i in [3, 4,5]:
    test_tensors(i)
