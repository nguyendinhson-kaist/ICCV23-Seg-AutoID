from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torch


x = torch.rand(1,3,1920,1080)
model = maskrcnn_resnet50_fpn()
print(model)
model.eval()
print(model(x))