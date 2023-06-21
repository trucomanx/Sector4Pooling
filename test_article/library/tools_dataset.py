#!/usr/bin/python
import sys
def load_dataset(dataset_name="mcfer2023"):
    if dataset_name=="mcfer2023":
        dataset_csv_train_file='/mnt/boveda/DATASETs/FACE-EMOTION/mcfer_v1.0/archive/train/training_labels.csv';
        dataset_csv_test_file ='/mnt/boveda/DATASETs/FACE-EMOTION/mcfer_v1.0/archive/test/test_labels.csv';
        dataset_train_base_dir='/mnt/boveda/DATASETs/FACE-EMOTION/mcfer_v1.0/archive/train'
        nout=7;
        input_shape=(224,224,3);
    elif dataset_name=="ck+48":
        dataset_csv_train_file='/mnt/boveda/DATASETs/FACE-EMOTION/CK+48/CK+48/labels.csv';
        dataset_csv_test_file ='/mnt/boveda/DATASETs/FACE-EMOTION/CK+48/CK+48/labels.csv';
        dataset_train_base_dir='/mnt/boveda/DATASETs/FACE-EMOTION/CK+48/CK+48'
        nout=5;
        input_shape=(96,96,3);
    else:
        sys.exit('Unknown dataset:',dataset_name);
        
    return dataset_csv_train_file, dataset_csv_test_file,dataset_train_base_dir,input_shape,nout;
