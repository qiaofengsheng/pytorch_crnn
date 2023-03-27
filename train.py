from tool.utils import *
from dataset.dataset import *
from torchvision import transforms
from model.model import *
from torch import nn,optim
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

is_use_cuda = True

if is_use_cuda:
    local_rank,device = use_ddp()
else:
    device = torch.device("cpu")

model = CRNN(32,3,28,512).to(device)

if is_use_cuda:
    model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)
    

train_transform = transforms.Compose([
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = CrnnDataset(
    image_path='/data/qfs/study_demo/crnn_pytorch/data/image',
    label_path='/data/qfs/study_demo/crnn_pytorch/data/label.txt',
    transform=train_transform,
    dict_path='/data/qfs/study_demo/crnn_pytorch/data/dicts.txt'
)

if is_use_cuda:
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,batch_size=64,sampler=train_sampler,collate_fn=train_dataset.collate_fn)
else:
    train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,collate_fn=train_dataset.collate_fn)

loss_fn = torch.nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters())

save_model_path = "/data/qfs/study_demo/crnn_pytorch/weights"
    
for i in range(5000):
    train_one_epoch(1,model,device,loss_fn,optimizer,train_loader,train_loader,save_model_path)