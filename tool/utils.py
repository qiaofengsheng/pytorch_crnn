import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm,os

def use_ddp():
    dist.init_process_group(
        backend="nccl",
        init_method='env://'
        )
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank,device


def train_one_epoch(epoch_id,model,device,loss_fn,optimizer,train_dataloader,val_dataloader,save_model_path):
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model.train()
    with tqdm.tqdm(train_dataloader) as t1:
        for image,label,label_len in tqdm.tqdm(train_dataloader):
            image,label = image.to(device),label.to(device)
            output = model(image)
            output_len = torch.full((output.shape[1],),output.shape[0],dtype=torch.long)
            train_loss = loss_fn(output,label,output_len,label_len)
            print(train_loss)
            # tqdm.tqdm.write('Epoch [{}]], Loss: {:.4f}'.format(epoch_id,train_loss.item()))
    #         t1.set_postfix(train_loss=train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
    model_without_ddp = model.module
    torch.save(model_without_ddp.state_dict(),os.path.join(save_model_path,f"crnn_epoch_{epoch_id}.pth"))
    # model.eval()
    # with torch.no_grad():
    #     with tqdm.tqdm(len(train_dataloader)) as t1:
    #         for image,label in train_dataloader:
    #             image,label = image.to(device),label.to(device)
    #             output = model(image)
    #             t1.set_description('Epoch %i' % epoch_id)
    #             val_loss = loss_fn(output,label)
    #             t1.set_postfix(val_loss=val_loss.item())



