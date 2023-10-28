# Prerequisites

MMDetection works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 9.2+ and PyTorch 1.8+.

Python 3.8.1, CUDA 11.7 and PyTorch 2.0.1 have been used to create this repository.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name s2fd python=3.8.1 -y
conda activate s2fd
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

```shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

# Installation

OpenMMLab recommends the following best practices to install MMDetection-based repositories.

## Best Practices

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

Note: mmcv 2.0.1 is used in this repository.

**Step 1.** Install the repository.

To install this repository from source:

```shell
git clone https://github.com/davideresta/sentinel2_field_detection.git
cd sentinel2_field_detection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Please refer to [MMDetection get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) if you need more information
about the installation and configuration of MMDetection.






# Dataset
The [AI4Boundaries dataset](https://data.jrc.ec.europa.eu/dataset/0e79ce5d-e4c8-4721-8773-59a4acf2c9c9) has been used in this repository.

The training images and their masks should be located in the data/sentinel2/train/ and in the data/sentinel2/train_masks/ folders, respectively.

Similarly, the validation images and their masks should be located in the data/sentinel2/val/ and in the data/sentinel2/val_masks/ folders.

Masks in the .tif format cannot be directly used by MMDetection models. Annotations in the Coco .json format are required.

To generate the annotations:

**Step 1.** Create a new folder named data/sentinel2/annotations.

**Step 2.** Move into the create_s2_annotations folder.

**Step 3.** Launch the create_s2_annotations.py script:

```shell
python create_s2_annotations.py
```

When the process is completed, two files named 'instances_train.json' and 'instances_val.json' should be in the data/sentinel2/annotations/ folder.

# Train a new model

This command can be used to train a Mask R-CNN + Swin model on the Ai4Boundaries dataset:

```shell
python tools\train.py configs\swin\mask-rcnn_swin-t-p4-w7_fpn_1x_sentinel2.py
```

The configuration file mask_rcnn_swin-t-p4-w7_fpn_1x_sentinel2.py was created starting from the mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py file.
Similarly, the configuration files of other models should be adapted to AI4Boundaries images before they can be used on them.

An input convolutional block is added to the considered model to make it suitable for 30-channel input tensors. Using an input block that reduces
the number of channels from 30 to 3 instead of directly adopting a (not pre-trained) 30-channel Swin backbone seems to make the training process faster and more effective.


# Test an existing model

To evaluate a previously trained field detection model, use the command:

```shell
python tools\test.py work_dirs\mask-rcnn_swin-t-p4-w7_fpn_1x_sentinel2\mask-rcnn_swin-t-p4-w7_fpn_1x_sentinel2.py work_dirs\mask-rcnn_swin-t-p4-w7_fpn_1x_sentinel2\epoch_15 --show
```

The first two arguments are, respectively, the path to the configuration file of the model and the path to the checkpoint file, containing the values of the trained parameters. The --show option can be used to visualize the predictions of the model on the considered test images.
