# Super Resolution - Machine Learning

[![Super Resolution Linux](https://github.com/voldien/SuperResolution/actions/workflows/ci.yaml/badge.svg)](https://github.com/voldien/SuperResolution/actions/workflows/ci.yaml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub release](https://img.shields.io/github/release/voldien/SuperResolution.svg)](https://github.com/voldien/SuperResolution/releases)

A SuperResolution training program for creating/training upscaling machine learning model, developed for educational purposes only. The result may vary between training data and hyperparameter, the example are from own trained model.

Project Was developed using **Python 3.9**

## Features

### Model Architecture

* **EDSR** - Enhanced Deep Residual Networks for Single Image Super-Resolution
* **VDR** - Very Deep Convolutional Network
* **AE** - AutoEncoder Super Resolution
* **DCNN** - Deep Convolutional Super Resolution Neural Network
* **Resnet** - Residual Network Deep Convolutional Super Resolution Neural Network
* **SRGAN** - GAN (Generative Adversarial Network) based Super Resolution Network

### Loss/Cost Function

* **SSIM** - Structural similarity index measure
* **MSA** - Mean Square Absolute
* **MSE** - Mean Square Error
* **VGG16** - Perceptible Loss Error Function
* **VGG19** - Perceptible Loss Error Function

## Basic Program Command Line

The most basic command line. Using the default option.

```bash
python superresolution/SuperResolution.py --data-set-directory /path_to_training_data/
```

### EDSR - Enhanced Deep Residual Networks for Single Image Super-Resolution

```basha
python superresolution/SuperResolution.py --batch-size 16 --epochs 10 --image-size 128 128 --model edsr --learning-rate 0.0003 --decay-rate 0.9 --decay-step 10000 --color-space rgb --loss-fn msa --shuffle-data-set-size 1024 --show-psnr --data-set-directory /path_to_training_data/   --output-dir image-super-resolution-result/
```

![Gangsta Anime EDSR Super Resolution Example from Trained model](https://github.com/voldien/SuperResolution/assets/9608088/1951a0c3-cebb-4ea8-818e-4a04bf28e116)
![Amagi Brilliant Park Anime EDSR Super Resolution Example from Trained model](https://github.com/voldien/SuperResolution/assets/9608088/17609c40-3b86-4a0d-a562-20d71359655a)

### VDR - Very Deep Convolutional Network

```bash
python superresolution/SuperResolution.py  --batch-size 16 --epochs 10 --image-size 128 128 --model vdr --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr --data-set-directory /path_to_training_data/ --output-dir image-super-resolution-result/
```

![Gangsta Anime EDSR Super Resolution Example from Trained model](https://github.com/voldien/SuperResolution/assets/9608088/24cccb38-807f-4454-bbc6-35ad9e03b57f)
![Amagi Brilliant Park Anime EDSR Super Resolution Example from Trained model](https://github.com/voldien/SuperResolution/assets/9608088/153792f5-c35a-4fae-8bba-aed47c8902de)

### AE - AutoEncoder Super Resolution

```bash
python superresolution/SuperResolution.py  --batch-size 16 --epochs 10 --image-size 128 128 --model dcsr-ae --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr --data-set-directory /path_to_training_data/ --output-dir image-super-resolution-result/
```

![Gangsta Anime EDSR Super Resolution Example from Trained model](https://github.com/voldien/SuperResolution/assets/9608088/0dac4554-6169-4662-9401-204feac33846)
![Amagi Brilliant Park Anime EDSR Super Resolution Example from Trained model](https://github.com/voldien/SuperResolution/assets/9608088/bc77e853-a5e8-4eac-880b-e6d7a5f3c801)

### DCNN - Deep Convolutional Super Resolution Neural Network

```bash
python superresolution/SuperResolution.py --batch-size 16 --epochs 10 --image-size 128 128 --model cnnsr --learning-rate 0.002 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr --data-set-directory /path_to_training_data/ --output-dir image-super-resolution-result/
```

![Gangsta Anime EDSR Super Resolution Example from Trained model](https://github.com/voldien/SuperResolution/assets/9608088/f164b778-296d-4ded-b658-ef46d8e77910)
![Amagi Brilliant Park Anime EDSR Super Resolution Example from Trained model](https://github.com/voldien/SuperResolution/assets/9608088/e5c33097-72ed-4c42-92a4-3a24d45b2110)

### Resnet - Residual Network Deep Convolutional Super Resolution Neural Network

```bash
python superresolution/SuperResolution.py --batch-size 16 --epochs 10 --image-size 128 128 --model dcsr-resnet --learning-rate 0.0003 --color-space rgb --loss-fn msa --shuffle-data-set-size 512 --show-psnr --data-set-directory /path_to_training_data/ --output-dir image-super-resolution-result/ 
```

### SuperResolution Training Program Argument

```bash
usage: SuperResolution [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--use-checkpoint] [--checkpoint-filepath CHECKPOINT_DIR] [--checkpoint-every-epoch CHECKPOINT_EVERY_NTH_EPOCH] [--learning-rate LEARNING_RATE] [--device DEVICES] [--cpu] [--gpu]
                       [--distribute-strategy {mirror}] [--verbosity VERBOSITY] [--use-float16] [--cache-ram] [--cache-file CACHE_PATH] [--shuffle-data-set-size DATASET_SHUFFLE_SIZE] [--data-set-directory TRAIN_DIRECTORY_PATHS]
                       [--validation-data-directory VALIDATION_DIRECTORY_PATHS] [--test-data-directory TEST_DIRECTORY_PATHS] [--image-size INPUT_IMAGE_SIZE INPUT_IMAGE_SIZE] [--output-image-size OUTPUT_IMAGE_SIZE OUTPUT_IMAGE_SIZE] [--seed SEED]
                       [--color-space {rgb,lab}] [--color-channels {1,3,4}] [--optimizer {adam,rmsprop,sgd,adadelta}] [--disable-validation] [--config CONFIG] [--model-filename MODEL_FILEPATH] [--output-dir OUTPUT_DIR] [--example-batch EXAMPLE_NTH_BATCH]
                       [--example-batch-grid-size EXAMPLE_NTH_BATCH_GRID_SIZE] [--show-psnr] [--metrics {psnr,ssim}] [--decay-rate LEARNING_RATE_DECAY] [--decay-step LEARNING_RATE_DECAY_STEP] [--model {dcsr,dscr-post,dscr-pre,edsr,dcsr-ae,dcsr-resnet,vdsr,srgan}]
                       [--loss-fn {mse,ssim,msa,psnr,vgg16,vgg19,none}]

Super Resolution Training Model Program

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Set the number of passes that the training set will be trained against.
  --batch-size BATCH_SIZE
                        number of training element per each batch, during training.
  --use-checkpoint      Set the path the checkpoint will be saved/loaded.
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
  --image-size INPUT_IMAGE_SIZE INPUT_IMAGE_SIZE
                        Set the size of the images in width and height for the model.
  --output-image-size OUTPUT_IMAGE_SIZE OUTPUT_IMAGE_SIZE
                        Set the size of the images in width and height for the model.
  --seed SEED           Set the random seed
  --color-space {rgb,lab}
                        Select Color Space used in the model.
  --color-channels {1,3,4}
                        Select Number of channels in the color space. GrayScale, RGB and RGBA.
  --optimizer {adam,rmsprop,sgd,adadelta}
                        Select optimizer to be used
  --disable-validation  Select if use data validation step.
  --config CONFIG       Config File - Json.
  --model-filename MODEL_FILEPATH
                        Define file path that the generator model will be saved at.
  --output-dir OUTPUT_DIR
                        Set the output directory that all the models and results will be stored at
  --example-batch EXAMPLE_NTH_BATCH
                        Set the number of train batches between saving work in progress result.
  --example-batch-grid-size EXAMPLE_NTH_BATCH_GRID_SIZE
                        Set the grid size of number of example images.
  --show-psnr           Set the grid size of number of example images.
  --metrics {psnr,ssim}
                        Set what metric to capture.
  --decay-rate LEARNING_RATE_DECAY
                        Set Learning rate Decay.
  --decay-step LEARNING_RATE_DECAY_STEP
                        Set Learning rate Decay Step.
  --model {dcsr,dscr-post,dscr-pre,edsr,dcsr-ae,dcsr-resnet,vdsr,srgan}
                        Set which model type to use.
  --loss-fn {mse,ssim,msa,psnr,vgg16,vgg19,none}
                        Set Loss Function
```

### Evolution Program - HyperParameter

The Evolution Program allow sto try multiple variable combination in order to find a good set of variable configuration. Similar to hyperparameter testing.

```bash
python3 superresolution/super-resolution-evolution-test.py  --epochs 8 --batch 32 rgb  --image-size 128 128  --data-set-directory /path_to_training_data/ --validation-data-directory /path_to_validation_data/   --output-dir evolution_test/
```

Argument options

```bash
usage: SuperResolution Model Evolution [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--use-checkpoint] [--checkpoint-filepath CHECKPOINT_DIR] [--checkpoint-every-epoch CHECKPOINT_EVERY_NTH_EPOCH] [--learning-rate LEARNING_RATE]
                                       [--device DEVICES] [--cpu] [--gpu] [--distribute-strategy {mirror}] [--verbosity VERBOSITY] [--use-float16] [--cache-ram] [--cache-file CACHE_PATH] [--shuffle-data-set-size DATASET_SHUFFLE_SIZE]
                                       [--data-set-directory TRAIN_DIRECTORY_PATHS] [--validation-data-directory VALIDATION_DIRECTORY_PATHS] [--test-data-directory TEST_DIRECTORY_PATHS] [--image-size INPUT_IMAGE_SIZE INPUT_IMAGE_SIZE]
                                       [--output-image-size OUTPUT_IMAGE_SIZE OUTPUT_IMAGE_SIZE] [--seed SEED] [--color-space {rgb,lab}] [--color-channels {1,3,4}] [--optimizer {adam,rmsprop,sgd,adadelta}] [--disable-validation] [--config CONFIG]
                                       [--output-dir OUTPUT_DIR] [--models [{cnnsr,dcsr,edsr,dcsr-ae,dcsr-resnet,vdsr,srgan,esrgan} ...]] [--loss-functions [{mse,ssim,msa,vgg16,vgg19} ...]] [--optimizer-evolution [{adam,rmsprop,sgd,adadelta} ...]]

Super Resolution Training Model Evolution Program

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Set the number of passes that the training set will be trained against.
  --batch-size BATCH_SIZE
                        number of training element per each batch, during training.
  --use-checkpoint      Set the path the checkpoint will be saved/loaded.
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
  --image-size INPUT_IMAGE_SIZE INPUT_IMAGE_SIZE
                        Set the input training images size. Low Resolution (LR).
  --output-image-size OUTPUT_IMAGE_SIZE OUTPUT_IMAGE_SIZE
                        Set the size of the images in width and height for the model (HR).
  --seed SEED           Set the random seed
  --color-space {rgb,lab}
                        Select Color Space used in the model.
  --color-channels {1,3,4}
                        Select Number of channels in the color space. GrayScale, RGB and RGBA.
  --optimizer {adam,rmsprop,sgd,adadelta}
                        Select optimizer to be used
  --disable-validation  Disable validation if validation data is present.
  --config CONFIG       Config File - Json.
  --output-dir OUTPUT_DIR
                        Set the output directory that all the models and results will be stored at
  --models [{cnnsr,dcsr,edsr,dcsr-ae,dcsr-resnet,vdsr,srgan,esrgan} ...]
                        Override what Model to include in training evolution.
  --loss-functions [{mse,ssim,msa,vgg16,vgg19} ...]
                        Override what Loss functions to include in training evolution.
  --optimizer-evolution [{adam,rmsprop,sgd,adadelta} ...]
                        Select optimizer to be used
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

### Upscale Videos

The following allows to upscale video.

```bash
python3 superresolution/UpScaleVideo.py --model super-resolution-model-2113109.h5 --save-output  high_video/ --input-file video_directory/ --batch 32 --color-space rgb
```

## Installation Instructions

### Setup Virtual Environment

python3.9 or higher

```bash
python3 -m venv venv
source venv/bin/activate
```

## Installing Required Packages

### CPU Only

```bash
pip install -r requirements.txt
```

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

## Convert Keras Model to TensorLite

```bash
python3 superresolution/generate_tflite.py --model super-resolution-model.keras --output model-lite.tflite
```

## Convert to ONNX

```bash
python3 -m tf2onnx.convert --saved-model tensorflow-model-path --output model.onnx
```

## Run Python In Background (Server)

When running python script as background process, it will still be terminated if closing the terminal window. However, with the **nohup** it can be run in the background as well close the terminal window.

```bash
nohup python3 superresolution/SuperResolution.py ...your arguments... &
```

## License

This project is licensed under the GPL+3 License - see the [LICENSE](LICENSE) file for details.
