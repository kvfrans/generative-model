# Brief Examination of Generative Models

Code for: https://kvfrans.com/a-brief-examination-of-generative-models/

To run this code, first install the requirements.
```
conda install -c anaconda jupyter seaborn
conda install -c conda-forge matplotlib opencv jax flax
conda install scipy
pip install einops clu tqdm orbax-checkpoint
```
If there are errors, use the versions below:
```
jax=0.4.14
flax=0.7.1
jaxlib=0.4.14
chex=0.1.82
optax=0.1.7
orbax-checkpoint=0.3.1
```

Then, run any of the python notebooks and the models will train and evaluate.
