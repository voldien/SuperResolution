# Super Resolution 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A SuperResolution training program for creating/training upscaling model, developed for educational purposes.

## Basic Program Command Line

### EDSR - Enhanced Deep Residual Networks for Single Image Super-Resolution

```bash
python superresolution/SuperResolution.py  --data-set-directory /path_to_training_data/ --batch-size 16 --epochs 10 --output-dir image-super-resolution-result/ --image-size 128 128 --model edsr --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr
```

### VDR - 

```bash
python superresolution/SuperResolution.py  --data-set-directory /path_to_training_data/ --batch-size 16 --epochs 10 --output-dir image-super-resolution-result/ --image-size 128 128 --model vdr --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr
```

### AE - AutoEncoder

```bash
python superresolution/SuperResolution.py  --data-set-directory /path_to_training_data/ --batch-size 16 --epochs 10 --output-dir image-super-resolution-result/ --image-size 128 128 --model dcsr-ae --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr
```

### DCNN - Deep Convolutional Neural Network

```bash
python superresolution/SuperResolution.py  --data-set-directory /path_to_training_data/ --batch-size 16 --epochs 10 --output-dir image-super-resolution-result/ --image-size 128 128 --model cnnsr --learning-rate 0.002 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr
```

### Resnet -

```bash
python superresolution/SuperResolution.py  --data-set-directory /path_to_training_data/ --batch-size 16 --epochs 10 --output-dir image-super-resolution-result/ --image-size 128 128 --model dcsr-resnet --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr
```

## Features

### Model Architecture
* **EDSR**
* **VDR**

### Loss/Cost Function
* **SSIM**
* **MSA**
* **MSE**


### SuperResolution Training Program Argument

```bash
usage: [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] --data-set-directory DATA_SETS_DIRECTORY_PATHS [--output-dir OUTPUT_DIR] [--image-size IMAGE_SIZE IMAGE_SIZE] [--output-image-size OUTPUT_IMAGE_SIZE OUTPUT_IMAGE_SIZE] [--seed SEED]
```

## Upscale Image

Upscaling images using pre-trained upscale model.

### Upscale Single Image
The following allows to upscale a single image.

```bash
python3 superresolution/UpScaleUtil.py --model super-resolution-model-2113109.h5 --input-file low_res.png --save-output  high_res.png --batch 32 --color-space rgb
```

### Upscale Directory
The following allows to upscale a whole directory.

```bash
python3 superresolution/UpScaleUtil.py --model super-resolution-model-2113109.h5 --save-output  high_output_dir/ --input-file low_input_dir/ --batch 32 --color-space rgb
```

### Evolution Program - HyperParameter
The Evolution Program allow sto try multiple variable combination in order to find a good set of variable configuration. Simliar to hyperparameter testing.

```bash
python3 superresolution/UpScaleUtil.py --model super-resolution-model-2113109.h5 --save-output  high_output_dir/ --input-file low_input_dir/ --batch 32 --color-space rgb
```


```bash
usage: [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] --data-set-directory DATA_SETS_DIRECTORY_PATHS [--output-dir OUTPUT_DIR] [--image-size IMAGE_SIZE IMAGE_SIZE] [--output-image-size OUTPUT_IMAGE_SIZE OUTPUT_IMAGE_SIZE] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Set the number of passes that the training set will be trained against.
  --batch-size BATCH_SIZE
                        number of training element per each batch, during training.
  --data-set-directory DATA_SETS_DIRECTORY_PATHS
                        Directory path where the images are located dataset images
  --output-dir OUTPUT_DIR
                        Set the output directory that all the models and results will be stored at
  --image-size IMAGE_SIZE IMAGE_SIZE
                        Set the size of the images in width and height for the model.
  --output-image-size OUTPUT_IMAGE_SIZE OUTPUT_IMAGE_SIZE
                        Set the size of the images in width and height for the model.
  --seed SEED           Seed
```

## Installation Instructions

### Setup Virtual Environment

### Installing Required Packages

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the GPL+3 License - see the [LICENSE](LICENSE) file for details.
