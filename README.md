# Crowd Flow Analysis using RAFT
This repository contains a project on optical flow estimation using the RAFT (Recurrent All-Pairs Field Transforms) algorithm. The goal is to estimate the motion of objects in crowd scenes using the CrowdFlow dataset.


## Dataset
To download the CrowdFlow dataset, please visit http://ftp01.nue.tu-berlin.de/crowdflow/TUBCrowdFlow.rar. 

```commandline
wget http://ftp01.nue.tu-berlin.de/crowdflow/TUBCrowdFlow.rar
sudo apt-get install unrar
unrar x TUBCrowdFlow
```

or use the following direct links 
- http://ftp01.nue.tu-berlin.de/crowdflow/TUBCrowdFlow.rar
- https://tubcloud.tu-berlin.de/s/FrDpjGfGJgPmHzN

Once downloaded, organize the dataset files as follows:

- Training Images: `/datasets/training/images`
- Training Ground Truth Flow: `/datasets/training/flow`
- Testing Images: `/datasets/testing/images`
- Testing Ground Truth Flow: `/datasets/testing/flow`

Make sure to place the dataset files in the respective directories before running the code.


## Quick run
- Set up dataset folder
- Run the notebook `run_raft_with_attention.ipynb` to perform short training with preloaded weights from our traning and visualize results on the `raft-attention` model.
- Run `plots.ipynb` to plot the train loss curve.


## Installation
To set up the project, follow the installation instructions below:

1. Clone the repository:
```commandline
git clone https://github.com/Ritika9494/ECE285-OpticalFlow.git
``` 

2. Install the libraries:
```commandline
pip install -r requirements.txt
```


## Running the Main Script
To run the main script with the provided options, use the following command:
```commandline
python src/main.py --name raft-basic --epochs 1000 --iters 12 --test False --checkpoint 0 --loss L1-loss
``` 

For the `--name` option, choose one of the following option:
- `raft-basic`: RAFT-S model
- `raft-attention`: RAFT-S with self attention in context encoder
- `raft-charbonnier`: RAFT-S with charbonnier loss
- `raft-attention-charbonnier`: : RAFT-S with self attention in context encoder

For the `--loss` option, choose either `L1-loss` or `charbonnier-loss`.

Set `--test` to True for only testing


## Results
The results of the optical flow estimation can be found in the `{model_name}\` folder. This folder contains the output images and intermediate model weights at regular intervals.


## Files Description
The following section gives an overview of the files in the repository and what they do.

- `src/main.py`: This file is responsible for initiating the training and testing process for each model. It takes command-line arguments to specify the model, number of epochs, iterations, testing mode, checkpoint, and loss function.

- `src/raft.py`: This file contains the implementation of the RAFT (Recurrent All-Pairs Field Transforms) algorithm. It includes the network architecture for RAFT and functions related to optical flow estimation.

- `src/trainer.py`: This file includes the trainer and testing functions for RAFT. It contains functions for computing the loss and metrics during training and testing.

- `src/dataloader.py`: This file provides utility functions for loading the CrowdFlow dataset. It handles data loading and preprocessing tasks.

- `src/Read_Write_Files.py`: This file contains utility functions to read and write optical flow files in the .flo format. It helps with handling flow data.

- `src/flow_viz.py`: This file includes utility functions for visualizing optical flow. It provides functions to visualize the flow field and overlay flow on input images.

- `plots.ipynb`: This notebook is used to generate plots using the saved `{model_name}/train_metric.npz` file. It plots training loss and end-point errors (EPE) during the training process.

- `run_raft_with_attention.ipynb`: This notebook performs a short training on preloaded weights and visualizes the results. It demonstrates the use of RAFT with self-attention in the context encoder.

- `download_pretrained_weights.sh`: This script can be used to download pre-trained RAFT models on the Flying Chairs and Flying Things datasets. It fetches the necessary weights for initialization.


## Contributors
1. Ritika Kishore Kumar
2. Surya Prakash Thoguluva Kumaran Babu
