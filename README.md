# Super Resolution - Machine Learning
[![Super Resolution Linux](https://github.com/voldien/SuperResolution/actions/workflows/ci.yaml/badge.svg)](https://github.com/voldien/SuperResolution/actions/workflows/ci.yaml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub release](https://img.shields.io/github/release/voldien/SuperResolution.svg)](https://github.com/voldien/SuperResolution/releases)

A SuperResolution training program for creating/training upscaling machine learning model, developed for educational purposes only.

## Features

### Model Architecture
* **EDSR** - Enhanced Deep Residual Networks for Single Image Super-Resolution
* **VDR** - Very Deep Convolutional Network
* **AE** - AutoEncoder Super Resolution
* **DCNN** - Deep Convolutional Super Resolution Neural Network
* **Resnet** - Residual Network Deep Convolutional Super Resolution Neural Network

### Loss/Cost Function
* **SSIM** - Structural similarity index measure
* **MSA** - Mean Square Absolute
* **MSE** - Mean Square Error
* **VGG16** - Perceptible Loss Error Function

## Basic Program Command Line

### EDSR - Enhanced Deep Residual Networks for Single Image Super-Resolution

```bash
python superresolution/SuperResolution.py --batch-size 16 --epochs 10 --image-size 128 128 --model edsr --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr --data-set-directory /path_to_training_data/   --output-dir image-super-resolution-result/
```

### VDR - Very Deep Convolutional Network

```bash
python superresolution/SuperResolution.py  --batch-size 16 --epochs 10 --image-size 128 128 --model vdr --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr --data-set-directory /path_to_training_data/ --output-dir image-super-resolution-result/
```

### AE - AutoEncoder Super Resolution

```bash
python superresolution/SuperResolution.py  --batch-size 16 --epochs 10 --image-size 128 128 --model dcsr-ae --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr --data-set-directory /path_to_training_data/ --output-dir image-super-resolution-result/
```

### DCNN - Deep Convolutional Super Resolution Neural Network

```bash
python superresolution/SuperResolution.py --batch-size 16 --epochs 10 --image-size 128 128 --model cnnsr --learning-rate 0.002 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr --data-set-directory /path_to_training_data/ --output-dir image-super-resolution-result/
```

### Resnet - Residual Network Deep Convolutional Super Resolution Neural Network

```bash
python superresolution/SuperResolution.py --batch-size 16 --epochs 10 --image-size 128 128 --model dcsr-resnet --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr --data-set-directory /path_to_training_data/ --output-dir image-super-resolution-result/ 
```

### SuperResolution Training Program Argument

```bash
usage: SuperResolution [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--checkpoint-filepath CHECKPOINT_DIR] [--checkpoint-every-epoch CHECKPOINT_EVERY_NTH_EPOCH] [--learning-rate LEARNING_RATE] [--device DEVICES] [--cpu] [--gpu] [--distribute-strategy {mirror}] [--verbosity VERBOSITY] [--use-float16]
                       [--cache-ram] [--cache-file CACHE_PATH] [--shuffle-data-set-size DATASET_SHUFFLE_SIZE] [--data-set-directory TRAIN_DIRECTORY_PATHS] [--validation-data-directory VALIDATION_DIRECTORY_PATHS] [--test-data-directory TEST_DIRECTORY_PATHS] [--image-size IMAGE_SIZE IMAGE_SIZE]
                       [--output-image-size OUTPUT_IMAGE_SIZE OUTPUT_IMAGE_SIZE] [--seed SEED] [--nr_image_example_generate NUM_EXAMPLES_TO_GENERATE] [--color-space {rgb,lab}] [--color-channels {1,3,4}] [--optimizer {adam,ada,rmsprop,sgd,adadelta}] [--disable-validation] [--model-filename MODEL_FILEPATH]
                       [--output-dir OUTPUT_DIR] [--example-batch EXAMPLE_BATCH] [--example-batch-grid-size width height] [--show-psnr] [--metrics METRICS] [--decay-rate LEARNING_RATE_DECAY] [--decay-step LEARNING_RATE_DECAY_STEP] [--model {cnnsr,dcsr,dscr-post,dscr-pre,edsr,dcsr-ae,dcsr-resnet,vdsr}]
                       [--loss-fn {mse,ssim,msa,psnr,vgg16,none}]

Super Resolution Training Model Program

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Set the number of passes that the training set will be trained against.
  --batch-size BATCH_SIZE
                        number of training element per each batch, during training.
  --checkpoint-filepath CHECKPOINT_DIR
                        Set the path the checkpoint will be saved/loaded.
  --checkpoint-every-epoch CHECKPOINT_EVERY_NTH_EPOCH
                        Set how often the checkpoint will be update, per epoch.
  --learning-rate LEARNING_RATE
                        Set the initial Learning Rate
  --device DEVICES      Select the device explicitly that will be used.
  --cpu                 Explicit use the CPU as the compute device.
  --gpu                 Explicit use of GPU
  --distribute-strategy {mirror}
                        Select Distribute Strategy.
  --verbosity VERBOSITY
                        Set the verbosity level of the program
  --use-float16         Hint the usage of Float 16 (FP16) in the model.
  --cache-ram           Use System Memory (RAM) as Cache storage.
  --cache-file CACHE_PATH
                        Set the cache file path that will be used to store dataset cached data.
  --shuffle-data-set-size DATASET_SHUFFLE_SIZE
                        Set the size of the shuffle buffer size, zero disables shuffling.
  --data-set-directory TRAIN_DIRECTORY_PATHS
                        Directory path where the images are located dataset images
  --validation-data-directory VALIDATION_DIRECTORY_PATHS
                        Directory path where the images are located dataset images
  --test-data-directory TEST_DIRECTORY_PATHS
                        Directory path where the images are located dataset images
  --image-size IMAGE_SIZE IMAGE_SIZE
                        Set the size of the images in width and height for the model.
  --output-image-size OUTPUT_IMAGE_SIZE OUTPUT_IMAGE_SIZE
                        Set the size of the images in width and height for the model.
  --seed SEED           Set the random seed
  --nr_image_example_generate NUM_EXAMPLES_TO_GENERATE
                        Number
  --color-space {rgb,lab}
                        Select Color Space used in the model.
  --color-channels {1,3,4}
                        Select Number of channels in the color space. GrayScale, RGB and RGBA.
  --optimizer {adam,ada,rmsprop,sgd,adadelta}
                        Select optimizer to be used
  --disable-validation  Select if use data validation step.
  --model-filename MODEL_FILEPATH
                        Define file path that the generator model will be saved at.
  --output-dir OUTPUT_DIR
                        Set the output directory that all the models and results will be stored at
  --example-batch EXAMPLE_BATCH
                        Set the number of train batches between saving work in progress result.
  --example-batch-grid-size width height
                        Set the grid size of number of example images.
  --show-psnr           Set the grid size of number of example images.
  --metrics METRICS     Set what metric to capture.
  --decay-rate LEARNING_RATE_DECAY
                        Set Learning rate Decay.
  --decay-step LEARNING_RATE_DECAY_STEP
                        Set Learning rate Decay Step.
  --model {cnnsr,dcsr,dscr-post,dscr-pre,edsr,dcsr-ae,dcsr-resnet,vdsr}
                        Set which model type to use.
  --loss-fn {mse,ssim,msa,psnr,vgg16,none}
                        Set Loss Function

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
python3 superresolution/super-resolution-evolution-test.py  --epochs 8 --batch 32 rgb  --image-size 128 128  --data-set-directory /path_to_training_data/ --validation-data-directory /path_to_validation_data/   --output-dir evolution_test/
```

Argument options
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

```bash
python3 -m venv venv
```

## Installing Required Packages

### Nvidia - CUDA
```bash
pip install -r requirements.txt requirements_cuda.txt
pip install tensorflow[and-cuda]==2.14.1
```

### AMD - ROCM
```bash
pip install -r requirements.txt requirements_rocm.txt
```

## Docker

### AMD - ROCM

```bash
docker build -t super-resolution-rocm -f Dockerfile.rocm .
docker run --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name sr-rocm super-resolution-rocm 
```
### Nvidia - CUDA

```bash
sudo apt-get install -y nvidia-container-toolkit
```

```bash
docker build -t super-resolution-cuda -f  Dockerfile.cuda .
docker run --network=host --gpus all --name sr-cuda super-resolution-cuda 
```


## License

This project is licensed under the GPL+3 License - see the [LICENSE](LICENSE) file for details.
