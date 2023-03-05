from torch.utils.data import Dataset,DataLoader
from .config import *
# from config import *
import cv2
import os
import torch
from torchvision import transforms
import numpy as np

class CrnnDataset(Dataset):
    def __init__(self,image_path=None,label_path=None,transform=None,dict_path=None) -> None:
        super().__init__()
        self.image_path = image_path
        self.data = open(label_path).readlines()
        self.transform = transform
        self.word2index_dict = get_dicts(dict_path)[0]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_name,label_str = self.data[index].replace("\n","").split("###")
        image = cv2.imread(os.path.join(self.image_path,img_name))
        h,w,c = image.shape
        new_w = int(100/32*h)
        padding_w = new_w - w%new_w
        mask = np.ones((h,padding_w,3))*128.0
        mask = mask.astype(np.uint8)
        image = np.concatenate([image,mask],axis=1)
        image = cv2.resize(image,(100,32))
        label=[]
        for i in label_str:
            label.append(int(self.word2index_dict[i]))
        image = self.transform(image)

        return image,torch.LongTensor(label) 

    def collate_fn(self,datas):
        image = []
        label = torch.LongTensor([])
        len_label=[]
        for data in datas:
            image.append(data[0])
            label=torch.concat([label,data[1]])
            len_label.append(len(data[1]))
        
        return torch.stack(image),label,torch.tensor(len_label)


from torchvision import transforms
train_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = CrnnDataset(
    image_path='../data/image',
    label_path='../data/label.txt',
    transform=train_transform,
    dict_path='../data/dicts.txt'
)
for i in train_dataset:
    print(i[0].shape,i[1])