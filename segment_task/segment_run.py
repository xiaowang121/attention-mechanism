import torch
from torch import nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.autograd import Variable
from NET_fcn import fcn,fcn_se,fcn_cbam,fcn_RLM,fcn_RLV

import numpy as np
from read_set_voc import image_transforms,VOCSegDataset
from read_set_coco import CocoDataset, coco_transforms

result_path = r'./coco_result_FCN_init.csv'

EPOCHES = 100
learn_rate = 0.0001
height = 224
width = 224
batch_size = 8

### voc dataset

# num_classes = 21
# train = VOCSegDataset(True, height, width, image_transforms)
# val = VOCSegDataset(False, height, width, image_transforms)
# test = VOCtest(height, width, image_transforms)
# train_data = DataLoader(train, batch_size=batch_size, shuffle=True)
# valid_data = DataLoader(val, batch_size=batch_size)
# test_data = DataLoader(test, batch_size=batch_size)

####  coco dataset

num_classes = 81
train = CocoDataset(True, height, width, coco_transforms)
val = CocoDataset(False, height, width, coco_transforms)

train_data = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_data = DataLoader(val, batch_size=batch_size)

def appendWrite(filename, content):
    with open(filename, 'a') as file_object:
        file_object.write(content)

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength = n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis = 1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis = 1) + hist.sum(axis = 0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis = 1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc
###fcn
net = fcn(num_classes)
# net = fcn_se(num_classes)
# net = fcn_cbam(num_classes)
# net = fcn_RLV(num_classes)
# net = fcn_RLM(num_classes)


net = net.cuda()

criterion = nn.CrossEntropyLoss()
basic_optim = torch.optim.Adam(net.parameters(), lr=learn_rate)
optimizer = basic_optim

def train_fcn(net,e):
        _train_loss = 0
        _train_acc = 0
        _train_acc_cls = 0
        _train_mean_iu = 0
        _train_fwavacc = 0

        net = net.train()
        for img_data, img_label in train_data:
            if torch.cuda.is_available:
                im = Variable(img_data).cuda()
                label = Variable(img_label).cuda()
            else:
                im = Variable(img_data)
                label = Variable(img_label)

            out = net(im)
            out = f.log_softmax(out, dim = 1)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _train_loss += loss.item()

            label_pred = out.max(dim = 1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()

            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                _train_acc += acc
                _train_acc_cls += acc_cls
                _train_mean_iu += mean_iu
                _train_fwavacc += fwavacc

        print('Epoch: {}, Train_Loss: {:.5f}, Train_Acc: {:.5f}, Train Acc_cls: {:.5f},Train Mean_IU: {:.5f},Train fwavacc:{:.5f}'.format(
            e+1, _train_loss / len(train_data), _train_acc / len(train),_train_acc_cls/len(train),
            _train_mean_iu / len(train), _train_fwavacc/ len(train)))
        appendWrite(result_path, '\n Epoch: {}, Train_Loss: {:.5f}, Train_Acc: {:.5f}, Train Acc_cls: {:.5f},Train Mean_IU: {:.5f},Train fwavacc:{:.5f}'.format(
            e+1, _train_loss / len(train_data), _train_acc / len(train),_train_acc_cls/len(train),
            _train_mean_iu / len(train), _train_fwavacc/len(train)))

        if (e) % 1 == 0:
            # torch.save(net.state_dict(), './train_models_voc/model%d.pth'%e)
            torch.save(net.state_dict(), './train_models_coco/model%d.pth' % e)
    #### valditon
        net = net.eval()

        _val_loss = 0
        _val_acc = 0
        _val_acc_cls = 0
        _val_mean_iu = 0
        _val_fwavacc = 0

        for img_data, img_label in valid_data:
            if torch.cuda.is_available():
                im = Variable(img_data).cuda()
                label = Variable(img_label).cuda()
            else:
                im = Variable(img_data)
                label = Variable(img_label)
            # forward
            with torch.no_grad():
                out = net(im)
            out = f.log_softmax(out, dim = 1)
            loss = criterion(out, label)
            _val_loss += loss.item()

            label_pred = out.max(dim = 1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                _val_acc += acc
                _val_acc_cls += acc_cls
                _val_mean_iu += mean_iu
                _val_fwavacc += fwavacc

        print('Epoch: {}, Valid_Loss: {:.5f}, Valid_Acc: {:.5f}, Valid_Acc_cls: {:.5f}, Valid_Mean IU: {:.5f},Valid_fwavacc: {:.5f} '.format(
            e+1, _val_loss / len(valid_data), _val_acc / len(val),_val_acc_cls/len(val),
            _val_mean_iu / len(val),_val_fwavacc/len(val)))
        appendWrite(result_path,
                    '\n Epoch: {}, Valid_Loss: {:.5f}, Valid_Acc: {:.5f}, Valid_Acc_cls: {:.5f}, Valid_Mean IU: {:.5f},Valid_fwavacc: {:.5f} '.format(
            e+1, _val_loss / len(valid_data), _val_acc / len(val),_val_acc_cls/len(val),
            _val_mean_iu / len(val),_val_fwavacc/len(val)))


if __name__ == '__main__':

    for epoch in range(EPOCHES):
        train_fcn(net,epoch)

        if epoch % 50 == 0:
            learn_rate *= 0.1
            optimizer.param_groups[0]['lr'] = learn_rate
