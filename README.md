# CSE-144: Applied Deep Learning Final Project
## Authors: Ashwin Senthilvasan & Ido Haiby
Final Project: Kaggle Transfer Learning Challenge for CSE 144: Applied Deep Learning @ UCSC (Spring 2026)

### Model Weights

The Google Drive link to a trained model (83% accuracy on test set) can be found here:
https://drive.google.com/file/d/1aDkKpv0v0jFD0irii3RDNWHShopNyao6/view?usp=sharing 

### Setup, Training, and Inference

Setup the Environment
```bash
# first install uv following this page:
# https://docs.astral.sh/uv/getting-started/installation/ 
# then clone the repo using git clone
$ cd cse-144-final-project
$ mkdir data
# download the dataset through kaggle or manual download
$ uv sync # create venv and sync dependencies 
```

Run the Training and/or Inference
```bash

# might have to change the PyTorch version to be compatible with whatever GPU you are using

$ uv run main.py # train the model (should output a model.pth)

$ uv run predict.py # inference on the test set (should output a submission.csv)
```

### Results on Kaggle Leaderboard

We achieve an 83.63% accuracy on the public Kaggle leaderboard.

![Train vs Validation Loss](kaggle_leaderboard.png)