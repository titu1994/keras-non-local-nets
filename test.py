'''A simple non-local block test'''
import numpy as np

from non_local import non_local_block

def test_tensors(rank=3):

    x = np.random.randint(1,5,(rank*[3]))*1.0

    print (x)

    x = non_local_block(x, compression=2, mode='embedded')
    print (x)

for i in [3, 4,5]:
    test_tensors(i)
