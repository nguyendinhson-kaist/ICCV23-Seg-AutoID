# ICCV23-Seg-AutoID

## Setup Environment

**Step1:** Create a conda environment with Python=3.9. Command:

```bash
conda create --name iccv23-py39
```

**Step2:** Install Pytorch with CUDA 11.8. Command:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step3:** Install mmdetection 3.1.0 (we will use it as a dependency). Command:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

Other packages can be install by pip or conda --forge. Please check requirements.txt for more detail

## Data preparation

**Step1:** download [here](https://drive.google.com/file/d/1wf_-XuGnrMO4-PEpxtbvGuzEzjAwsgUy/view?usp=drive_link) and extract dataset as follow:

```text
data
├── train
│   ├── image1.png
│   └── ...
├── val
│   ├── image1.png
│   └── ...
├── test
│   ├── image1.png
│   └── ...
├── train.json
├── val.json
└── test.txt
```

**Step2 (optional):** Because mmdet supports both RLE format and Binary format, you can convert RLE annotation to Polygon annotation. Run bellow command:

```bash
python utils/rle2polygon.py data/train.json data/poly_train.json
python utils/rle2polygon.py data/val.json data/poly_val.json
```

Your datafolder should be like this:

```text
data
├── train
│   ├── image1.png
│   └── ...
├── val
│   ├── image1.png
│   └── ...
├── test
│   ├── image1.png
│   └── ...
├── train.json
├── val.json
├── poly_train.json
├── poly_val.json
└── test.txt
```

Note that we add prefix "poly_" by convention which matchs with the dataset config (change annotation file names in config file).

**Step3 (optional):** we also use a specialized CopyPaste augmentation technique. To train models with our CopyPaste technique, you need to prepare a folder that contains all ground-truth instances from trainning set. Run the bellow command:

```bash
python utils/extract_objects.py data_root
```

A **cropped_objects** folder will be created inside the data folder.

**Step4 (optional):** In case you want to visualize the augmented images, we provide a tool to help you. Run the below command:

```bash
python data_config --output-dir data_sample --not-show
```

## Learn about configs

### Get ready with mmdet config

Everything in mmdet is about config. Before starting with mmdet, you should read their tutorial first. mmdetection (or mmdet) is built on top of a core package called mmengine. I highly recommend you to check their homepage and github for detail documentation and tutorials.

- Github: <https://github.com/open-mmlab/mmengine>

- Hompage: <https://mmengine.readthedocs.io/en/latest/get_started/introduction.html>

Or read their config explanation at least:

- Github: <https://github.com/open-mmlab/mmengine/tree/main/docs/en/tutorials>

### Our configs

Our dataset config can be found at:

```text
./configs/_base_/datasets/vipriors_instance.py
```

Mask R-CNN model configs:

```text
./configs/exp/mask_rcnn
```

## Train a model

Create a config for your experiment and save in ./configs folder. Then run below command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py config_path model_name
```

Training outputs (checkpoints, logs, merged config, tensorboard log, etc.) will be available in ./output/(model_name) folder

## Export submission result

After training a model, you can export submission result. Submission can be produced on val set or test set (if val set, you can see the evalution score). Run below command:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py config_path checkpoint_path --valid --show-dir predicted_imgs_folder_name
```

If you produce submission on test set, omit --valid argument.

If you don't want to export predicted images, omit --show-dir. Note that --show-dir only accepts folder name, not folder dir (I use --show-dir to match with mmdet convention). The config_path and checkpoint_path should be in ./output folder.

After running the test command, you can find the inference result as "result.segm.json". Rename it to "submission.json" then zip it. Now it is ready to submit to the test server.
