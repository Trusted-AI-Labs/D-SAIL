# D-SAIL - Distributed Secure AI Learning

![D-SAIL](docs/img/d-sail_line.png)

Welcome to the D-SAIL's Github repository!

[![Documentation Status](https://readthedocs.org/projects/d-sail/badge/?version=latest)](https://d-sail.readthedocs.io/en/latest/?badge=latest)

The documentation of the code is available on [readthedocs](https://d-sail.readthedocs.io/en/latest/)

## Description

In short, this repository currently hosts code to pseudonymized or anonymized medical images in the DICOM format and train models using PyTorch in a distributed fashion, using PyTorch. 

The general framework encompasses tools allowing the training of a global model from multiple sub-models. The main advantage is that the participants do not exchange their local data while benefiting from a presumably higher performing model. In exchange, the partners need to train a similar model on their local data. Particularly, in federated learning, a consortium of actors share the weights of their locally trained model and a central unit aggregates the latter. Although it allows the partners to only share their model's gradients and weights, the architecture also raises several challenges to ensure a privacy-preserving system.  First, the pseudonymization of the training dataset. Secondly, the confidentiality of the models and the gradients has to be guaranteed to prevent any reverse engineering to the training dataset. Eventually, the protection of the model against degradation by training on inadequate data.  

## Data Preparation

0. Clone the repository

```
git clone https://github.com/Trusted-AI-Labs/D-SAIL.git
cd ResidualsTracking
```

1. Create the environment with Anaconda

```
conda env create -n d-sail python=3.7
conda activate d-sail
pip install -r requirements.txt
```

2. Create the data folders

```
mkdir data
mkdir data/output
mkdir data/input
mkdir data/input/class0
mkdir data/input/class1
...
```

Next, copy all your DICOM files in the appropriate folders.

3. Call the script to add the labels in the DICOM files

Check the path to your Python interpreter of your Anaconda environment, then run the command with your own path.

```
E:/Anaconda3/envs/d-sail/python.exe dicom_converter/add_metadata.py path/to/input_folder/ 'value' '[tag]'
```

For instance:
```
E:/Anaconda3/envs/d-sail/python.exe dicom_converter/add_metadata.py ~/d-sail/data/input/class1/ '1' '[0x0014, 0x2016]'
```

4. Copy all files from different classes and place it together in the same folder:

```
mkdir data/input/all
cp -R data/input/class0/* data/input/all
cp -R data/input/class1/* data/input/all
...
```

5. Pseudonimization & Hashing of DICOM files:

```
E:/Anaconda3/envs/d-sail/python.exe dicom_pseudonymizer/anonymizer.py data/input/all data/input/all-pseudo --lookup=path/to/lookup_table.csv
```

For instance:

```
E:/Anaconda3/envs/d-sail/python.exe dicom_pseudonymizer/anonymizer.py path/to/input_folder path/to/output_folder --lookup=path/to/lookup_table.csv
```

If you wish to have you files renamed with the pseudonimized information, add the `--renameFiles` option

6. Decompose DICOM files to PNG and JSON files

```
E:/Anaconda3/envs/d-sail/python.exe dicom_converter/utils/dicom_to_img.py path/to/input_folder path/to/output_folder
```

7. Classify the data in different class folders 

```
E:/Anaconda3/envs/d-sail/python.exe dicom_converter/classify_data.py path/to/input_folder '[tag]' path/to/output_folder
```

For instance:
```
E:/Anaconda3/envs/d-sail/python.exe dicom_converter/classify_data.py data/input/all-pseudo '[tag]' data/output
```

8. Divide the data in train/valid/test folders:

For instance:

```
E:/Anaconda3/envs/d-sail/python.exe dicom_converter/utils/cat_to_dataset.py /Database/Raw/ /Database/H/ --percentages [0.7,0.2,0.1]
```

9. (Optional) Split the data in multiple datasets, e.g. to get one for each hospital:

For instance:

```
E:/Anaconda3/envs/d-sail/python.exe dicom_converter/utils/hospital_split.py /Database/H/ /Database/ --percentages [0.5,0.3,0.2]
```

## Federated Learning

### Train on local machine

1. Activate the environment

```
conda activate d-sail
```

2. Launch the server node

```
python federated_learning/server/server.py
```

3a. Launch the client nodes (3x, or depending on the number of partners) with the appropriate data split path and number of epochs

```
python federated_learning/client/H_federated.py --split "50_33_17" --db_loc "Hospitals" --db "cancer" --res_loc "results" --hospital "H0" --resize 50
```

3b. To train solely on local data, use `H_nofederated` script instead, also in the client folder: 
```
python federated_learning/client/H_nofederated.py --split "50_33_17" --db_loc "Hospitals" --db "cancer" --res_loc "results" --hospital "H0" --resize 50
```

Note: You can use the following command to see the complete set of parameters available:
```
python federated_learning/client/H_federated.py -h
python federated_learning/client/H_nofederated.py -h
```

## References

The code to pseudonimize the DICOM files was adapted from https://github.com/KitwareMedical/dicom-anonymizer, please refer to their repository for details on initial implementation.

The federated learning framework used is https://github.com/adap/flower, please refer to their repository for details on implementation.
