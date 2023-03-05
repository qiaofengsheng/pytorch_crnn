from model.model import CRNN
import cv2
import torch
from dataset.config import *
from torchvision import transforms
import numpy as np
import os



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRNN(32,3,28,512).to(device)
    model.load_state_dict(torch.load("./weights/crnn_epoch_1.pth",map_location=device))

    for im in os.listdir("./data/test"):

        image = cv2.imread(os.path.join("./data/test",im))
        h,w,c = image.shape
        new_w = int(100/32*h)
        padding_w = new_w - w%new_w
        mask = np.ones((h,padding_w,3))*128.0
        mask = mask.astype(np.uint8)
        image = np.concatenate([image,mask],axis=1)
        image = cv2.resize(image,(100,32))


        # transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        image = image/255.0

        # image = transform(image).unsqueeze(dim=0).to(device)
        image =torch.Tensor(image).permute(2,0,1).unsqueeze(dim=0).to(device)
        output = model(image)
        output = output.squeeze()
        out = torch.argmax(output,dim=1).cpu()

        word2index_dicts,index2word_dicts=get_dicts("./data/dicts.txt")

        char_list = []
        for i in range(len(out)):
            if out[i] != 0 and (not (i > 0 and out[i - 1] == out[i])):
                char_list.append(out[i])
        s=""
        for i in char_list:
            s+=index2word_dicts[i.item()]
        print(im,s)


