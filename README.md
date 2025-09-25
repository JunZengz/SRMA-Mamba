# SRMA-Mamba: Spatial Reverse Mamba Attention Network for Pathological Liver Segmentation in MRI Volumes

## Overview


## Create Environment
```
conda create -n SRMA-Mamba python==3.9.0
conda activate SRMA-Mamba
```

## Install Dependencies
```    
pip install -r requirements.txt
```

## Download Dataset
Download CirrMRI600+ T1W and T2W dataset from [this link](https://osf.io/cuk24/files/osfstorage). Move it to the `data` directory.

## Train
```
python train.py
```

## Test
```
python test.py
```

## Weight Files 
Our weight files and result maps are available on [Google Drive](https://drive.google.com/file/d/1F9TWv2zOz9ny0L8SJ8IeSo7ODhYDrV06/view?usp=drive_link).


## Citation
Please cite our paper if you find the work useful:
```
@article{zeng2025srma,
  title={SRMA-Mamba: Spatial Reverse Mamba Attention Network for Pathological Liver Segmentation in MRI Volumes},
  author={Zeng, Jun and Huang, Yannan and Keles, Elif and Aktas, Halil Ertugrul and Durak, Gorkem and Tomar, Nikhil Kumar and Trinh, Quoc-Huy and Nayak, Deepak Ranjan and Bagci, Ulas and Jha, Debesh},
  journal={arXiv preprint arXiv:2508.12410},
  year={2025}
}
```

## Contact
Please contact zeng.cqupt@gamil.com for any further questions.
