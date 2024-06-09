# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# import torchvision
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
#
# from collections import namedtuple
# from sklearn.metrics import classification_report
#
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
#
import torch
torch.PYTORCH_NO_CUDA_MEMORY_CACHING=1
torch.PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb
x = torch.rand(5, 3)
print(x)
