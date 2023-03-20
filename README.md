# FLGAN
Conditional Generative Adversarial Network for predicting radiotherapy-induced ventilation change

## Create and activate the Conda environment:
  ```
  conda env create -f environment.yml
  conda activate mlpackage
  ```
  
## Predict ventilation change using trained model:
  ```
  python predict.py
  ```
## Other files that live here:
  1. mlsubmit.sh & train.job
      - SGE scheduler and submission scripts for training from scratch
  2. submitpredict.sh & predict.job
      - SGE scheduler and submission scripts for prediction
  3. ganmodel/model_183920.h5
      - Pretrained model
