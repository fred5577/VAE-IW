# RolloutIW

Official code for:

[Planning From Pixels in Atari With Learned Symbolic Representations](https://arxiv.org/abs/2012.09126) (AAAI 2021)<br/>
Andrea Dittadi\*, Frederik K. Drachmann\*, Thomas Bolander


## Experiments

We compare our algorithm with RA Rollout IW using B-PROST featuers to show we outperform the previous efforts even though we greatly reduced the number of features we plan based on.

## Code structure
|File/Folder    | Description             |
|--------------|--------------------------|
|srC/screen.cpp|Code for calculating B-PROST features from the atari screen. pybind11 is used to import the C++ code into python|
|vae/models.py | Discrete VAE models.|
|vae/models_norm.py | Gaussian VAE models.|
| vae/loader.py | Loading images for training VAE's. |
| vae/utils.py | Different utility functions for the VAE models.|
| vae/plothistogram.py | Method used for plotting the distribution of 0 and 1 using a VAE model. |
| planner.py | THe main program used to run Rollout IW using B-PROST or VAE features. (also used for collecting images)|
| training.py | The program for training discrete VAE's.|
| training_norm.py | The program for training Gaussian VAE.|
| rolloutIW.py | The implementation of RolloutIW from (Hector Geffner) |
| IW | The implementation of IW. (not thoroughly tested) |
| sample.py | The method for choosing actions when we have build a tree using Rollout IW. (risk averse) |
| screen.py | The method that converts an image to binary using a VAE or B-PROST. |
| tree.py | A utility class that makes it easier to keep track of tree structure. |
| atari_wrapper.py | Wraps the atari environment, changing the frame skip and overwriting the get image class. |
| setup.py CMakeList.txt | For making the c++ files into python library. |
| util (folder) | Files that is used for drawing figures and plots. |


## Running code

Start by setting up the C++ library with the B-PROST. It is necessary to have regardless of running VAE or B-PROST features.

```python
pip install ./RolloutIW
```

### Run using B-RPOST

The planner can then be used to run with Rollout IW with B-PROST for 5 rounds and a 0.5 budget.

```python
python3 planner.py --env Alien-v4 --frame-skip 15 --time-budget 0.5 --round 5
```


### Run with VAE

1. If you already have an Atari dataset to train on move the next step. Otherwise run the following code to create a dataset of 15.000 images by running B-PROST

```python
python3 planner.py --env Alien-v4 --frame-skip 15 --time-budget 0.5 --save-images True
```

2. Train a model for Alien.

```python
python3 training.py --epochs 100 --batch-size 64 --env Alien --zdim 20 --sigma 1 --beta 0.0001 --image-training-size 128 --temp 0.5 --file-naming color_images --loss BCE --kernel-size 15 --seed 1
```

3. Run the planner with the VAE model.

```python
python3 planner.py --env Alien-v4 --image-size-planning 128 --xydim 15 --model-name {path to model} --test-round-of-model True --features model --zdim 20 --time-budget 0.5 --rounds-to-run 5
```

