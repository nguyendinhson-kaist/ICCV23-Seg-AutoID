import torchvision.transforms as transform
import torch
import numpy as np
import random

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

from pycocotools import mask as coco_mask
from detectron2.data import DatasetCatalog, MetadataCatalog

def visualize(dataset_name, num_samples=10, show_labels=True):
    '''
    Visualize some sample from a dataset with mask annotations at random

    Parameters:
        dataset_name: the name of the registered dataset
        num_samples: the number of samples to plot
        show_labels: whether show the category of each instance or not

    Returns:
        None 
    '''
    data_dict = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    random_samples = random.sample(data_dict, num_samples)

    fig, ax_subs = plt.subplots(nrows=num_samples, ncols=2)

    if num_samples == 1:
        ax_subs = (ax_subs,)

    for i, sample in enumerate(random_samples):
        ax1, ax2 = ax_subs[i]
        image_path = sample['file_name']
        image = Image.open(image_path)
        annotations = sample['annotations']

        ax1.imshow(image)

        colors = matplotlib.colormaps['hsv']

        blank_mask = np.array(image) / 255.0
        for i, anno in enumerate(annotations):
            rle_mask = anno['segmentation']
            bbox = anno['bbox']
            category_id = anno['category_id']
            category_name = metadata.get('thing_classes')[category_id]
            color = np.array(colors(i/len(annotations))[:3])[None, None, :]

            mask = coco_mask.decode(rle_mask)
            blank_mask = np.where(mask[:, :, None], blank_mask * 0.7 + color * 0.3, blank_mask)
            if show_labels:
                x, y, w, h = bbox
                ax2.text(x, y + h + 10, category_name, color='r', fontsize='small')

        ax2.imshow(blank_mask)

        ax1.axis('off')
        ax2.axis('off')

    plt.show()

def dataset_analysis(dataset_name):
    '''
    Get distribution of the dataset

    Parameters:
        dataset_name: the name of the registered dataset

    Returns:
        A dict contains statistic info of the dataset
    '''
    data_dict = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    num_images = len(data_dict)
    sizes = {}

    category_names = metadata.get('thing_classes')
    category_counts = [0] * len(category_names)

    for sample in data_dict:
        wh = '{}_{}'.format(sample['width'], sample['height'])
        if wh not in sizes:
            sizes[wh] = 0
        sizes[wh] += 1

        annotations = sample['annotations']
        for anno in annotations:
            category_counts[anno['category_id']] += 1

    results = {
        'dataset': dataset_name,
        'num_images': num_images,
        'categories': {},
        'sizes': sizes
    }

    for category_id, num in enumerate(category_counts):
        results['categories'][category_names[category_id]] = num

    return results