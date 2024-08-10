# Residual-INR: Communication Efficient Fog Online Learning Using Implicit Neural Representation

## Introduction

This repository includes all of the code implementation for the Residual-INR submitted to ICCAD'24.

## Environment Setup

- **Software**: All of our experiments are finihsed with Python. You can set up the Python environment by running the following command:

```bash
pip install -r requirements.txt
```

- **Hardware**: INR decoding on GPU is finished on NVIDIA RTX A6000 (48GB). You can run our code on any GPU platforms that can support PyTorch.

## Residual-INR Encoding




#### Background INR Encoding

If background INR encoding is based on **Rapid-INR** using MLPs. Run the following commands to encode a dataset:

```bash
cd B_INR_encode
python background_INR_encoding_Rapid_INR.py --dataset_dir ../data/OTB/
```
The dataset path is specified as the --dataset_dir flag. 

If background INR encoding is based on **NeRV** using mixed MLPs and CNN. Run the following commands to encode a dataset:

```bash
cd B_INR_encode
python background_INR_encoding_NeRV.py --train_image_path ../data/OTB/
```

#### Generate Object Residual


```bash
cd O_INR_encode
python generate_residual.py ---recon_npy_path ./recon_object_array --ref_npy_path ./raw_object_array --residual_output_path ./residual_array
```

#### Object INR Encoding

Object INR encoding is based on tiny MLPs. Run the following commands to encode a dataset:

```bash
cd O_INR_encode
python O_INR_encoding.py --area_array_path ../area_info --train_list_path ./residual_array --train_ref_list_path ./raw_object_array --train_recon_list_path ./B_INR_recon_object_array
```
The object area information is specified as the --area_array_path flag, the genrated object residual array is specified as the --train_list_path flag, the cropped raw object array is specified as the --train_ref_list_path flag, the INR reconstructed object array is specified as the --train_recon_list_path flag


