import os
import torch
from model.net_arch import get_net
from dataset.dataloder import get_test_dataloader
import cv2
import numpy as np
from TMQI.TMQI import TMQI

def test_model(dir_name):
    model_name = 'model/g_epoch100.pt'
    save_path = dir_name + 'test_result/LoLTestSet'
    os.mkdir(save_path)

    generator= get_net('TSTMNet')
    generator.load_state_dict(torch.load(dir_name + model_name))
    generator = generator.cuda()
    generator.eval()

    test_dataloader = get_test_dataloader()

    total_Q, total_S, total_N = 0, 0, 0

    for i, minibatch in enumerate(test_dataloader):
        if i == 2:
            break
        hdr_v, hdr, raw_hdr, ldr = minibatch
        hdr_for_TMQI = raw_hdr[0].numpy()
        ldr = ldr[0].numpy() 
        hdr_for_TMQI = np.transpose(hdr_for_TMQI, (1, 2, 0))
        hdr_v = hdr_v.cuda()
        hdr = hdr.cuda()

        with torch.no_grad():
            mapped_hdr_v = generator(hdr_v)


        mapped_hdr_v = mapped_hdr_v[0, 0] 
        hdr_h = hdr[0, 0]
        hdr_s = hdr[0, 1]
        mapped_hdr = torch.stack([hdr_h, hdr_s*0.8, mapped_hdr_v], dim=0)

        mapped_hdr = mapped_hdr.cpu()
        mapped_hdr = mapped_hdr.detach().numpy()
        mapped_hdr = np.transpose(mapped_hdr, (1, 2, 0))
        output = cv2.cvtColor(mapped_hdr, cv2.COLOR_HSV2RGB)
        Q, S, N, s_local, s_maps = TMQI()(hdr_for_TMQI, output*255)
        print(f"{i+1} TMQI  Q：{Q:.6f}, S：{S:.6f}, N：{N:.6f}")
        total_N += N
        total_Q += Q
        total_S += S
        cv2.imwrite(save_path + '/' + f"{i+1}.png", output*255)
        torch.cuda.empty_cache()
    print(f"avg TMQI Q：{total_Q/(i+1):.6f}, S：{total_S/(i+1):.6f}, N：{total_N/(i+1):.6f}")