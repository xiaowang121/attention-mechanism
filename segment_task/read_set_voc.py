import torch
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

import numpy as np
import os
from PIL import Image

voc_root = r'./VOC2007'
test_root = r'./VOC2007'
def read_images(root=voc_root, train=True):

    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in images]
    return data, label

def read_imagestest(root=test_root):

    txt_name = root + '/ImageSets/Segmentation/' + 'test.txt'
    with open(txt_name, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in images]
    return data, label

def crop(data, label, height, width):

    box = (0, 0, width, height)
    data = data.crop(box)
    label = label.crop(box)
    return data, label

classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]
cm2lbl = np.zeros(256**3)

for i, cm in enumerate(colormap):
    cm2lbl[(cm[0]*256 + cm[1])*256 + cm[2]] = i

def image2label(im):
    data = np.array(im, dtype="int32")
    idx = (data[:,:,0]*256 + data[:,:,1])*256 + data[:,:,2]
    return np.array(cm2lbl[idx], dtype="int64")

def image_transforms(data, label, height, width):
    data, label = crop(data, label, height, width)
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = im_tfs(data)
    label = image2label(label)
    label = torch.from_numpy(label)
    return data, label

class VOCSegDataset(torch.utils.data.Dataset):


    def __init__(self, train, height, width, transforms):
        self.height = height
        self.width = width
        self.fnum = 0
        self.transforms = transforms
        data_list, label_list = read_images(train = train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        if (train == True):
            print("train_set：loading " + str(len(self.data_list)) + " img and label")
        else:
            print("valid_set：loading " + str(len(self.data_list)) + " img and label")

    def _filter(self, images):
        img = []
        for im in images:
            img.append(im)
        return img

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.height, self.width)
        return img, label

    def __len__(self):
        return len(self.data_list)

class VOCtest(torch.utils.data.Dataset):
    def __init__(self, height, width, transforms):
        self.height = height
        self.width = width
        self.fnum = 0  
        self.transforms = transforms
        data_list, label_list = read_imagestest()
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print("test_set：loading " + str(len(self.data_list)) + " img and label" )

    def _filter(self, images):
        img = []
        for im in images:
            img.append(im)
        return img

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.height, self.width)
        return img, label

    def __len__(self):
        return len(self.data_list)



