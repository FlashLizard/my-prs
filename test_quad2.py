import torch
from data_handle.prs_dataset import PrsDataSet
from torch.utils.data import DataLoader
import argparse
from model.prs_loss  import sym_quad_tran,PrsLoss
import matplotlib.pyplot as plt
import numpy as np


def parse_arg():
    parse = argparse.ArgumentParser(description='test-process')
    parse.add_argument('--path', type=str, help='model path')
    parse.add_argument('--data_path', type=str, default='data_handle/results/output0.npz')
    
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arg()
    path = args.path
    data_path = args.data_path
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PrsDataSet(data_path, device)
    print('len',len(dataset))
    test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = torch.load(path)
    
    model.eval()
    for i, (voxels, sps, cps) in enumerate(test_loader):
        voxels_u = voxels.unsqueeze(1)
        voxels = voxels.to(device)
        sps = sps.to(device)
        cps = cps.to(device)
        with torch.no_grad():
            _, quads = model(voxels_u)
        quads.squeeze_(2)
        cp = cps[0].cpu().numpy()
        sp = sps[0].cpu().numpy()
        quad = quads[0].cpu().numpy().astype(float)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        ax.scatter(sp[...,0],sp[...,1],sp[...,2])
        tps = sym_quad_tran(sps,quads).cpu()
        loss = PrsLoss()
        
        id=0
        for w,i,j,k in quad:
            if(id != 0):
                id += 1
                continue
            w = 0
            vector = np.array([i,j,k])
            norm = np.linalg.norm(vector)
            vector /= norm
            i,j,k = vector
            print(w,i,j,k)
            # print(torch.stack((torch.from_numpy(xx),torch.from_numpy(yy),torch.from_numpy(zz)),dim=-1))
            ax.plot([0,i],[0,j],[0,k])
            ax.scatter(tps[0,id,:,0],tps[0,id,:,1],tps[0,id,:,2], alpha=0.1)
            loss_v = loss(voxels[0].unsqueeze(0),
                          sps[0].unsqueeze(0),
                          cps[0].unsqueeze(0),0,
                          quads[0,id].unsqueeze(0).unsqueeze(0)
                          )
            print(loss_v)
            id += 1
            break
        
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        
        if(input('continue?(y/n)')=='n'): break
    
        
    
