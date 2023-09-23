# Deep Unfolded D-ADMM
## Proper colored network with P = 50 agents
![P=50 graph](https://github.com/yoav1131/Deep-Unfolded-D-ADMM/assets/61379895/27bada02-b87a-432d-8817-011b7c59b950)

## Unfolded D-ADMM for LASSO at agent p in iteration k. Dashed green and blue blocks are the primal and dual updates, respectively. Red fonts represent trainable parameters
![update_step(3)](https://github.com/yoav1131/Deep-Unfolded-D-ADMM/assets/61379895/40ff6d9a-eb57-460f-9167-ef356df5df3b)

## Unfolded D-ADMM for linear regression model illustration at agent p in iteration k. Dashed green and blue blocks are the primal update and the dual update, respectively. Red fonts represent trainable parameters
![d-lr primal dual update](https://github.com/yoav1131/Deep-Unfolded-D-ADMM/assets/61379895/3ebb0ed9-82ff-4516-829c-d4d97a7a54d3)

## Introduction
In this work we propose a method that solves disributed optimization problem called Unfolded Distibuted Method of Multipliers(D-ADMM), which enables D-ADMM to operate reliably with a predefined and small number of messages exchanged by each agent using the emerging deep unfolding methodology. 
Unfolded D-ADMM fully preserves the operation of D-ADMM, while leveraging data to tune the hyperparameters of each iteration of the algorithm. 

Please refer to our [paper](https://github.com/yoav1131/Deep-Unfolded-D-ADMM/files/12705750/paper.pdf) for more detailes.

## Usage
This code has been tested on Python 3.9.7, PyTorch 1.10.2 and CUDA 11.1

### Prerequisite
* scipy
* tqdm
* numpy
* pytorch: https://pytorch.org
* matplotlib
* networknx
* TensorboardX: https://github.com/lanpa/tensorboardX

### Training
Example with 50 agents:

'''
python dlasso.py --exp_name dlasso_with_50_agents --data simulated --batch_size 100 --P 50 --graph_prob 0.12 --case dlasso --model diff --valid True
'''

or

'''
python dlr.py --exp_name dlasso_with_50_agents --data simulated --batch_size 100 --P 50 --graph_prob 0.12 --case dlasso --model diff --valid True
'''

### Testing
Example with 50 agents:

'''
python dlasso.py --exp_name dlasso_with_50_agents --eval --valid False
'''

or 

'''
python dlr.py --exp_name dlasso_with_50_agents --eval --valid False

'''

# Data
### Distributed LASSO Problem
Please refer to the  [data](https://drive.google.com/drive/folders/1fbPHrS1ICw4bvawPwJJNCiqBUjdLrDx2?usp=sharing) for the distributed LASSO problem.

The folder contains four directories for different SNR values {-2, 0, 2, 4}, in each directory there is a dataset_{snr}_snr.npy file which contain the data and labels. 

When you load the data set allow_pickle=True.

### Distributed Linear Regression Problem
For the distributed linear regression problem I used MNIST dataset.
