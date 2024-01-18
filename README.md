# Super Resolution 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A SuperResolution training program for creating upscaling machine model.

## Basic Program Command Line

### EDSR - 
```bash
python superresolution/SuperResolution.py  --data-set-directory /path_to_training_data/ --batch-size 16 --epochs 10 --output-dir image-super-resolution-result/ --image-size 128 128 --model edsr --learning-rate 0.0003 --color-space lab --loss-fn msa --shuffle-data-set-size 512
```

### VDR - 
```bash
python superresolution/SuperResolution.py  --data-set-directory /path_to_training_data/ --batch-size 16 --epochs 10 --output-dir image-super-resolution-result/ --image-size 128 128 --model vdr --learning-rate 0.0003 --color-space lab --loss-fn msa --shuffle-data-set-size 512
```

### AE - 
```bash
python superresolution/SuperResolution.py  --data-set-directory /path_to_training_data/ --batch-size 16 --epochs 10 --output-dir image-super-resolution-result/ --image-size 128 128 --model vdr --learning-rate 0.0003 --color-space lab --loss-fn msa --shuffle-data-set-size 512
```


### DCNN -


### Resnet -

## Upscale Image

Upscaling images using trained upscale model.

```bash
python3 superresolution/UpScaleUtil.py --model super-resolution-model-2113109.h5 --save-output  high_res.png --input-file low_res.png --batch 32 --color-space lab
```


## Installation Instructions

### Setup Virtual Environment

### Installing Required Packages

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the GPL+3 License - see the [LICENSE](LICENSE) file for details.
