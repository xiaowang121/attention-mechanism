from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import time
import NetWork_ResNet

log_path = r'./ResNet_se_44.txt'
model_path = r'./ResNet_se_44.pth'


def appendWrite(filename, content):
    with open(filename, 'a') as file_object:
        file_object.write(content)

def test(net, testloader):
    net.eval()
    incorrect = 0
    for i, data in enumerate(testloader):
        img, label = data
        img = img.cuda()
        label = label.cuda()
        with torch.no_grad():
            preds = net(img)
        predicted = torch.argmax(preds, 1)  
        
        incorrect += (predicted != label).sum().item()
    nTotal = len(testloader.dataset)
    err = incorrect / nTotal
    return err

# normMean = [0.49139968, 0.48215827, 0.44653124]
# normStd = [0.24703233, 0.24348505, 0.26158768]
normMean = [0.485, 0.456, 0.406]
normStd = [0.229, 0.224, 0.225]
normTransform = transforms.Normalize(normMean, normStd)

trainTransform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normTransform
])
testTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

trainloader = DataLoader(
    dset.CIFAR100(root=r'G:\computer-vision\cifar-100-python\cifar-100-python', train=True, download=False,
                 transform=trainTransform),
    batch_size=128, shuffle=True)
testloader = DataLoader(
    dset.CIFAR100(root=r'G:\computer-vision\cifar-100-python\cifar-100-python', train=False, download=False,
                 transform=testTransform),
    batch_size=128, shuffle=False)

net = NetWork_ResNet.init_resnet32()


if torch.cuda.is_available():
    net = nn.DataParallel(net).cuda()

criterion = nn.CrossEntropyLoss()
#
start = time.time()
for epoch in range(200):
    if epoch < 100:
        lr = 1e-1
    elif epoch == 100:
        lr = 1e-2
    elif epoch == 150:
        lr = 1e-3
    #lr=0.001
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for i, data in enumerate(trainloader):
        incorrect = 0
        total = 0
        img, label = data
        img = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        predict = net(img)

        loss = criterion(predict, label)
        loss.backward()
        optimizer.step()
        predicted = torch.argmax(predict, 1)  

        incorrect += (predicted != label).sum().item()
        total += label.size(0)

        if (i + 1) % 100 == 0:
            err = test(net, testloader)

            print('Epoch[{}/{}],batch:{},loss:{:.6f},train_error:{:.6f},test_error:{:.6f}'.format(epoch + 1, 1, i + 1, loss.item(), incorrect/total,err))
            appendWrite(log_path, '\tEpoch[{}/{}],batch:{},loss:{:.6f},train_error:{:.6f},test_error:{:.6f}\n'.format(epoch + 1, 1, i + 1, loss.item(), incorrect/total,err))
            save_path = model_path
            torch.save(net.state_dict(), save_path)
            net.train()

end = time.time()
delt = end-start
save_path = model_path
torch.save(net.state_dict(), save_path)

net.eval()
test_loss = 0
incorrect = 0
start_t = time.time()
for i, data in enumerate(testloader):
    img, label = data
    img = img.cuda()
    label = label.cuda()
    with torch.no_grad():
        preds = net(img)
    predicted = torch.argmax(preds, 1)    

    incorrect += (predicted != label).sum().item()
end_t = time.time()
delt_t = end_t-start_t
nTotal = len(testloader.dataset)
err = incorrect/nTotal
print('testerror：{}'.format(err))
print('训练时间', delt)
print('测试时间', delt_t)
appendWrite(log_path, '\ttesterror:{}\n'.format(err))
appendWrite(log_path, '\t训练时间:{}\n'.format(delt))
appendWrite(log_path, '\t测试时间:{}\n'.format(delt_t))