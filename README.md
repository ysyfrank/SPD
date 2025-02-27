# Overview
This is the project for Model-based IR.

### Requirements
1. Refer to /pretrain/requirements.txt.
   
### Run Models
##### Data preparation
```
cd starter_script
sudo bash prepare_data.sh
```

##### Train model
```
sudo bash train.sh  # change the parameters in train_t5.py for different data and stages
```

##### Test model
```
sudo bash test.sh
```
