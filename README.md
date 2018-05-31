# Keras Non-Local Neural Networks
Keras implementation of Non-local blocks from the paper ["Non-local Neural Networks"](https://arxiv.org/abs/1711.07971)

- Support for `"Gaussian"`, `"Embedded Gaussian"` and `"Dot"` instantiations of the Non-Local block. 
- Support for variable shielded computation mode (reduces computation by N**2 x, where N is default to 2)
- Support for `"Concatenation"` instantiation will be supported when authors release their code.

# Usage

The script `non_local.py` contains a single function : `non_local_block` which takes in an input tensor and wraps a non-local block around it.

```python
from non_local import non_local_block

ip = Input(shape=(##)) # this can be a rank 3, 4 or 5 tensor shape
x = ConvND(...)  # as againm can be Conv1D, Conv2D or Conv3D
x = non_local_block(x, compression=2, mode='embedded')
...

```

# Basic block
From the paper, a basic Non-Local block looks like below (with the Embedded Gaussian instantiation)

<img src="https://github.com/titu1994/keras-non-local-nets/blob/master/images/non-local-block.PNG?raw=true" width=100% height=100%>
