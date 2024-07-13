import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split



class PrsDataSet(Dataset):
    def __init__(self, data_path, device='cpu',max_id=1000):
        voxel_girds = []
        sample_point_sets = []
        closest_point_sets = []
        id = 0
        if(data_path[-4:]=='.npz'):
            print(data_path)
            data_part = np.load(data_path)
            print(data_part)
            voxel_girds.extend(data_part['voxel_grid'])
            sample_point_sets.extend(data_part['sample_points'])
            closest_point_sets.extend(data_part['closest_points'])
        else:
            while True:
                try:
                    print(f'{data_path}/output{id}.npz')
                    data_part = np.load(f'{data_path}/output{id}.npz')
                    print(data_part)
                    voxel_girds.extend(data_part['voxel_grid'])
                    sample_point_sets.extend(data_part['sample_points'])
                    closest_point_sets.extend(data_part['closest_points'])
                except Exception as e:
                    print('error',e)
                    print(f'Load {id} parts of data.')
                    break
                if id >= max_id:
                    break
                id += 1
        self.voxel_girds = torch.from_numpy(np.stack(voxel_girds)).float().to(device)
        self.sample_point_sets = torch.from_numpy(np.stack(sample_point_sets)).float().to(device)
        self.closest_point_sets = torch.from_numpy(np.stack(closest_point_sets)).float().to(device)
    
    def __len__(self):
        return len(self.voxel_girds)
    
    def __getitem__(self, idx):
        return self.voxel_girds[idx], self.sample_point_sets[idx], self.closest_point_sets[idx]

if __name__ == '__main__':
    data_path = 'results'
    dataset = PrsDataSet(data_path)
    print('len',len(dataset))
    test_size = min(len(dataset), 10)
    train_dataset, test_dataset = random_split(dataset, [test_size, len(dataset) - test_size])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    for i, (models, sps, cps) in enumerate(test_loader):
        print(models.shape, sps.shape, cps.shape)