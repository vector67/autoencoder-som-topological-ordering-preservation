# autoencoder-som-topological-ordering-preservation
This is the code used to work out whether or not topological ordering is preserved by an autoencoder

To run, you have to compile the pyx file first 

```python3 compile.py build_ext --inplace```

Then run 

```python mnist-dataset.py```

There are a whole bunch of depenencies required. The following is an incomplete list of them
* tensorflow 2.0
* matplotlib
* cython
* minisom
* numpy