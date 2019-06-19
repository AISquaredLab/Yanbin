
# coding:utf8

import os
from PIL import Image
from torch.utils import data
import numpy as numpy
from torchvision import transforms as T

class DogCat(data.Dataset):
    
    def __init__(self,root,transforms=None,train=True,test=False):
        #获取所有图片地址并划分数据
        self.test = test
        imgs = [os.path.join(root,img) for img in os.listdir(root)]

        #test1: data/test1/8973.jpg
        #train: data/train/cat.10004.jpg

        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1])) #test数据不要猫狗前缀
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

            imgs_num = len(imgs)

            #划分训练验证集
            if self.test:
                self.imgs = imgs
            elif train:
                self.imgs = imgs[int(0.7*imgs_num)]
            else:
                self.imgs = imgs[int(0.7*imgs_num)]

            if transforms is None:
                #测试，训练，验证的数据有区别
                normalize = T.Normalize(
                    mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])

                #测试集，验证集
                if self.test or not train:
                    self.transforms = T.Compose([
                        T.Scale(224),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        normalize
                        ])
                #训练集
                else:
                    self.transforms = T.Compose([
                        T.Scale(256),
                        T.RandomSizedCrop(224),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        normalize
                        ])

    #多线程加速
    def __getitem__(self,index):
        #如果测试集，不会告诉你是猫是狗
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)

        return data, label

    def __len__(self):
        return len(self.imgs)
        




