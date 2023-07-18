# Requirements
Apart from other libs already installed in primary environment. This project requires to install:
- submitit==1.4.5
```
conda install -c conda-forge submitit=1.4.5
```
- timm==0.3.2
```
pip install timm==0.3.2
```

# Data
## Pretrained data
Pretrained data is ICCV basketball dataset.
## Linear Probing data
Store linear probing data in `linprobe_data/` foler, which contains crop images for 3 classes: human, background, ball. Download [here](https://drive.google.com/file/d/1MDaCL-i9gUhp6IEaC6dxwU6DIK0Mimr4/view?usp=drive_link)

```
linprobe_data/train
linprobe_data/val
```
# Usage
## Pretrain
```
bash pretrain.sh
```

## Linear Probe
```
bash linprobe.sh
```

# Fix Bug
Intiate running may encounter these problems
```
# Bug
File "/home/daoduyhung/anaconda3/envs/iccv/lib/python3.9/site-packages/timm/models/layers/helpers.py", line 6, in <module>
    from torch._six import container_abcs

# Fix: modify helper.py file
import collections.abc as container_abcs

# Bug
File "mae/util/misc.py", line 21, in <module>
    from torch._six import inf
ModuleNotFoundError: No module named 'torch._six'

# Fix: change to
from torch import inf
```

# Linear Probing Experiment
- Loss: Cross Entropy
- Evaluation metric: F1 score, accuracy each class
## Masked AE
Freeze MAE pretrained on original dataset and finetune Linear layer on classification datasets obtained

|F1 score|Background Accuracy|Ball Accuracy|Human Accuracy|
|--------|-------------------|-------------|--------------|
|74.94   |44/46              |22/25        |314/379       |

## Resnet18
### Pretrained on ImageNet
Freeze Resnet pretrained on ImageNet dataset and finetune Linear layer on classification datasets obtained

|F1 score|Background Accuracy|Ball Accuracy|Human Accuracy|
|--------|-------------------|-------------|--------------|
|34.53   |3/46               |0/25         |376/379       |

### Train from scrash on evaluation data
|F1 score|Background Accuracy|Ball Accuracy|Human Accuracy|
|--------|-------------------|-------------|--------------|
|98.44   |44/46              |24/25        |379/379       |
