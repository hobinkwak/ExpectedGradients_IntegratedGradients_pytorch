## Expected / IntegratedGradients by PyTorch

### Overview
- Simple implementation of IntegratedGradients and ExpectedGradients by PyTorch
- Integrated Gradients is Feature Attribution methods for Neural Networks (Sundararajan et al., 2017)
    - [paper](https://arxiv.org/abs/1703.01365)
- Expected Gradients is an extension of IG, which samples baseline inputs from the given dataset (Erion et al., 2019)
    - [paper](https://arxiv.org/abs/1906.10670)

### Usage
```python
from attr import integrated_gradients, expected_gradients

model = ...
target = ...
n_iter = ...
baseline = ...
data = ...

ig_attr = integrated_gradients(model, data, target, n_iter)
eg_attr = expected_gradients(model, data, baseline, target, n_iter)
```
- n_iter : the number of iterations used by the approximation method
    - the higher n_iter is, the more accurate approximation but more memory usage

### Model and Data
- Model
    - simple 2 CNN layers
- Data
    - used MNIST data

### requirements
```
numpy
torch
matplotlib
```