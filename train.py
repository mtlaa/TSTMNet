import os
import torch
import torch.nn as nn
from model.net_arch import get_net
from dataset.dataloder import get_train_dataloader
import torchvision
from utils.log import get_log
import datetime
from utils.test import start_test
from torch.optim import lr_scheduler


result_dir = f"{datetime.datetime.now()}"
result_dir = result_dir[:-7]
result_dir = "result/" + result_dir + '/'
os.mkdir(result_dir)   

image_save = result_dir + "images/"
os.mkdir(image_save)   

model_save = result_dir + "model/"
os.mkdir(model_save)  

test_save = result_dir + "test_result/"
os.mkdir(test_save) 

log_path = result_dir + "info.log"
log_path1 = result_dir + "result.log"
logger = get_log(log_path, 'train')
logger1 = get_log(log_path1, 'test')

device = 'cuda:0'
use_gpu = torch.cuda.is_available()
logger.info(f"[{torch.cuda.device_count()}] gpu is avaliable")
if use_gpu:
    logger.info(f"use gpu:{torch.cuda.get_device_name(device)} for training")



batch_size = 1
dataloader = get_train_dataloader(batch_size=batch_size, num_works=16)
net = get_net('TSTMNet')
net = net.to(device)

lambda_pixel = 100
lambda_ganloss = 10
lambda_featloss = 10

lr = 0.0001
num_epoch = 100

optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0001)
scheduler = lr_scheduler.LinearLR(optimizer, 1, 0, num_epoch-30)
pixel_loss = nn.L1Loss().to(device)  


# ------------------------------------------------------------------------
#                               train
# ------------------------------------------------------------------------
logger.info(f"learning rate: {lr}")
logger1.info(f"learning rate: {lr}")
logger.info(f"num_epoch = {num_epoch} , start train........")

for epoch in range(1, num_epoch+1):
    min_loss = 100
    for i, minibatch in enumerate(dataloader):
        hdr, gt_ldr = minibatch
        hdr = hdr.to(device)
        gt_ldr = gt_ldr.to(device)
        
        pred_ldr = net(hdr)

        optimizer.zero_grad()
        loss = pixel_loss(pred_ldr, gt_ldr)

        loss.backward()
        optimizer.step()

        if (epoch % 10 == 0 | epoch == 1) & i == 0:
            for index, image in enumerate(pred_ldr):
                torchvision.utils.save_image(image, image_save + f"ldr_v_numepoch:{epoch}_{index}.png")


        if i % 100 == 0:
            for index, image in enumerate(pred_ldr):
                torchvision.utils.save_image(image, result_dir + f"1_ldrv.png")
            logger.info(f"epoch:{epoch}, loss:{loss.item():.6f}")

            if epoch==num_epoch:
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    g_path = model_save + f"g_epoch{num_epoch}.pt"
                    torch.save(net.state_dict(), g_path)
                    logger.info(f"saved model Generator as [g_epoch{num_epoch}.pt],g_loss={loss.item():.9f}")
                    logger1.info(f"saved model Generator as [g_epoch{num_epoch}.pt],g_loss={loss.item():.9f}")
        

    
    if epoch > 30:
        scheduler.step()
        logger.info(f"adjust learning rate to {scheduler.get_last_lr()[0]:.9f}")
    
    if epoch in (30, 40, 50, 60, 70, 80, 90, 110, 130, 150, 170, 190):
        logger.info(f"---------------epoch:{epoch} test---------------")
        epoch_test_save = test_save + f"epoch-{epoch}_test/"
        os.mkdir(epoch_test_save)
        start_test(epoch_test_save, net, logger)


logger1.info("---------------start final test---------------")
os.mkdir(test_save + 'final/')
start_test(test_save + 'final/', net, logger1)
logger1.info("----------------end final test----------------")
