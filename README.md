# Deeplearning Project Submission Fall 2023
by
Pranav Nedunghat (pn2187), 
Suraj Kiron Nair (skn8716),
Ritvik Nair (pn2187)


## Instructions
Install requirements
```
pip install -r requirements.txt
```
Setup resource folder  and download COCO dataset
```
./setup.sh
```
Create hdf5 files
```
python hdf5generator.py
```
Train the model:
  You can Tune hyperparameters in config.py
```
python train.py
```
Test the model
```
python test.py -W <best weight folder>
```
To perform inference 
```
python inference.py
```


