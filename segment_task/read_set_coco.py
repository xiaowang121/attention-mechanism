import torch
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image

voc_root = "G://coco//"
def read_images(root=voc_root, train=True):
    txt_fname = root + ('simple_train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'Images', i+'.jpg') for i in images]
    label = [os.path.join(root, 'Labels', i+'.png') for i in images]
    return data, label

def read_imagestest(root=voc_root):
    txt_fname = root +'test.txt'
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'Images', i+'.jpg') for i in images]
    label = [os.path.join(root, 'Labels', i+'.png') for i in images]
    return data, label

def crop(data, label, height, width):

    box = (0, 0, width, height)
    data = data.crop(box)
    label = label.crop(box)
    return data, label

classes = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
            'traffic light','fire hydrant','stop sign','parking meter', 'bench','bird',
            'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
            'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
            'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
            'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
            'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
            'cake','chair','couch','potted plant','bed','dining table','toilet','tv',
            'laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
            'refrigerator','book','clock','vase','scissors','teddy bear',
             'hair drier','toothbrush']

colormap = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],
            [0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0],[192,128,0],
            [64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128],[128,64,128],[0,192,128],[128,192,128],
            [64,64,0],[192,64,0],[64,192,0],[192,192,0],[64,64,128],[192,64,128],
            [64,192,128],[192,192,128],[0,0,64],[128,0,64],[0,128,64],[128,128,64],
            [0,0,192],[128,0,192],[0,128,192],[128,128,192],[64,0,64],[192,0,64],
            [64,128,64],[192,128,64],[64,0,192],[192,0,192],[64,128,192],[192,128,192],
            [0,64,64],[128,64,64],[0,192,64],[128,192,64],[0,64,192],[128,64,192],
            [0,192,192],[128,192,192],[64,64,64],[192,64,64],[64,192,64],[192,192,64],
            [64,64,192],[192,64,192],[64,192,192],[192,192,192],[32,0,0],[160,0,0],
            [32,128,0],[160,128,0],[32,0,128],[160,0,128],[32,128,128],[160,128,128],
            [96,0,0],[224,0,0],[96,128,0],[224,128,0],[96,0,128],[224,0,128],[96,128,128],
            [224,128,128],[32,64,0]]

cm2lbl = np.zeros(256**3)


for i, cm in enumerate(colormap):
    cm2lbl[(cm[0]*256 + cm[1])*256 + cm[2]] = i


def image2label(im):
    data = np.array(im, dtype="int32")
    idx = (data[:,:,0]*256 + data[:,:,1])*256 + data[:,:,2]
    return np.array(cm2lbl[idx], dtype="int64")

def coco_transforms(data, label, height, width):
    data, label = crop(data, label, height, width)

    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.471, 0.448, 0.408], [0.234, 0.239, 0.242])
    ])
    data = im_tfs(data)
    label = image2label(label)
    label = torch.from_numpy(label)
    return data, label


class CocoDataset(torch.utils.data.Dataset):
    # 构造函数
    def __init__(self, train, height, width, transforms):
        self.height = height
        self.width = width
        self.transforms = transforms
        data_list, label_list = read_images(train = train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        if (train == True):
            print("train_set：loadig " + str(len(self.data_list)) + " img and label")
        else:
            print("valid_set：loadig " + str(len(self.data_list)) + " img and label")
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

class Cocotest(torch.utils.data.Dataset):
    # 构造函数
    def __init__(self, height, width, transforms):
        self.height = height
        self.width = width
        self.transforms = transforms
        data_list, label_list = read_imagestest()
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print("test_set：loadig " + str(len(self.data_list)) + " img and label")

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

