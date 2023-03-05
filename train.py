from tool.utils import *
from dataset.dataset import *
from torchvision import transforms
from model.model import *
from torch import nn,optim
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'


local_rank,device = use_ddp()

model = CRNN(32,3,28,512).to(device)
model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)


train_transform = transforms.Compose([
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = CrnnDataset(
    image_path='./data/image',
    label_path='./data/label.txt',
    transform=train_transform,
    dict_path='./data/dicts.txt'
)

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(dataset=train_dataset,batch_size=64,sampler=train_sampler,collate_fn=train_dataset.collate_fn)

loss_fn = torch.nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters())

save_model_path = "./weights"
    
for i in range(5000):
    train_one_epoch(1,model,device,loss_fn,optimizer,train_loader,train_loader,save_model_path)