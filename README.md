# Keras Non-Local Neural Networks
Keras implementation of Non-local blocks from [[1]](https://arxiv.org/abs/1711.07971).

- Support for `"Gaussian"`, `"Embedded Gaussian"` and `"Dot"` instantiations of the Non-Local block. 
- Support for variable shielded computation mode (reduces computation by N**2 x, where N is default to 2)
- Support for `"Concatenation"` instantiation will be supported when authors release their code.

# Usage Templates

The script `non_local.py` contains the `NonLocalBlock` instance which takes in an input tensor and wraps a non-local block around it.

```python
from non_local import NonLocalBlock
from tensorflow.keras.layers import Input, Conv1D, Conv2D, Conv3D

ip = Input(shape=(...))  # input tensor with an "N" rank order of 3, 4 or 5
x = ConvND(...)         # convolution operation with aforementioned rank 
...
non_local_block = NonLocalBlock(intermediate_dim=None, compression=2, mode='embedded', add_residual=True)
x = non_local_block(x)
...
```

The script `non_local_layerstyle.py` contains the `NonLocalBlock` **layer** which takes in an input tensor and wraps a non-local block around it. Made to facilitate the neural network builder using the Sequential method. 

```python
from non_local_layerstyle import NonLocalBlock
from tensorflow.keras.layers import Input, Conv1D, Conv2D, Conv3D

# Define the input shape
input_shape = (...)  # shape of input tensor

model = Sequential()
model.add(ConvND(...))         # convolution operation with an "N" rank order of 3, 4 or 5
...
model.add(NonLocalBlock(intermediate_dim=None, compression=2, mode='embedded', add_residual=True))
...
```


# Basic block
From [[1]](https://arxiv.org/abs/1711.07971), a basic Non-Local block with the Embedded Gaussian instantiation has the below logic:



<center><img src="https://github.com/titu1994/keras-non-local-nets/blob/master/images/non-local-block.PNG?raw=true" width=50% ></center>


1. Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He. "Non-local Neural Networks." arXiv:1711.07971 [cs.CV], 21 Nov 2017. [Link](https://arxiv.org/abs/1711.07971)

