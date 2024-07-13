import torch
from data_handle.prs_dataset import PrsDataSet
from model.prs_model import PrsModel
from model.prs_loss  import PrsLoss
from torch.utils.data import DataLoader, random_split
import argparse

def parse_arg():
    parse = argparse.ArgumentParser(description='train-process')
    parse.add_argument('--begin_epoch', type=int, help='begin epoch', default=0)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arg()
    begin_epoch = args.begin_epoch
    
    data_path = 'data_handle/results'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PrsDataSet(data_path, device,1)
    print('len',len(dataset))
    test_size = int(len(dataset)*0.1)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    if(begin_epoch==0):
        model = PrsModel().to(device)
    else:
        model = torch.load('trained_model/model{begin_epoch}.pth')
    loss = PrsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 300
    min_epoch = 0
    min_loss = 1000000
    
    for epoch in range(epochs):
        model.train()
        for i, (voxels, sps, cps) in enumerate(train_loader):
            voxels_u = voxels.unsqueeze(1)
            voxels = voxels.to(device)
            sps = sps.to(device)
            cps = cps.to(device)
            optimizer.zero_grad()
            planes, quads = model(voxels_u)
            planes.squeeze_(2)
            quads.squeeze_(2)
            # print(voxels.shape,sps.shape,cps.shape, planes.shape, quads.shape)
            
            loss_value = loss(voxels, sps, cps, planes, quads)
            loss_value.backward()
            optimizer.step()
            # print(f'epoch{epoch} step{i}', loss_value)
        
        model.eval()
        loss_value = 0
        for i, (voxels, sps, cps) in enumerate(test_loader):
            voxels_u = voxels.unsqueeze(1)
            voxels = voxels.to(device)
            sps = sps.to(device)
            cps = cps.to(device)
            planes, quads = model(voxels_u)
            planes.squeeze_(2)
            quads.squeeze_(2)
            loss_value += loss(voxels, sps, cps, planes, quads)
        torch.save(model, f'trained_model/model{epoch}.pth')
        if(min_loss>loss_value):
            min_loss = loss_value
            min_epoch = epoch
        print(f'epoch{epoch} loss', loss_value)
    
    print(f'min_epoch{min_epoch}, loss{min_loss}')
        
    
