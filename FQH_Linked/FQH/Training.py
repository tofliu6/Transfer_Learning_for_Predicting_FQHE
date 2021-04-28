import os
import sys
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from Built_Model import Transformer

def get_path(path):
    file_list = [root+'/'+f for root, dirs, files in os.walk(path) for f in files]
    return file_list

def get_data(indir, outdir1, outdir2, outdir3):
    #outdir1 for Datasets, outdir2 for train_datasets, outdir3 for val_datasets
    paths = get_path(indir)
    l = np.load(paths[0]).shape[0]
    print(l)
    temp = []
    for path in paths:
        if int(path.split('/')[-1][0]) == 9:
            labels = np.load(path)
        else: temp.append(path)
    for npy in temp:
        final_npy = np.zeros((1, (np.load(npy).shape[1])+15))
        print(final_npy.shape[1])
        for i in range(l):
            temp_npy = np.load(npy)[i,...]
            temp_npy = np.hstack((temp_npy, labels[i,...]))
            final_npy = np.vstack((final_npy, temp_npy))
        final_npy = np.delete(final_npy, 0, axis=0)
        if not os.path.exists(outdir1):
            os.makedirs(outdir1)
        np.save(outdir1+'/'+npy.split('/')[-1][0]+'_ele_predata',final_npy)
        split_idx = int(final_npy.shape[0] * 0.8)
        np.random.shuffle(final_npy)
        train_data_npy = final_npy[:split_idx,...]
        if not os.path.exists(outdir2):
            os.makedirs(outdir2)
        np.save(outdir2 + '/' + npy.split('/')[-1][0] + '_train', train_data_npy)
        val_data_npy = final_npy[split_idx:,...]
        if not os.path.exists(outdir3):
            os.makedirs(outdir3)
        np.save(outdir3+'/'+npy.split('/')[-1][0]+'_val',val_data_npy)


def get_data_2(indir, outdir):
    paths = get_path(indir)
    l = np.load(paths[0]).shape[0]
    print(l)
    print(paths[0])
    for i in range(4,9):
        for j in range(len(paths)):
            if int(paths[j].split('/')[-1][0]) == i:
                temp_npy1 = np.load(paths[j])
                print(paths[j])
            if int(paths[j].split('/')[-1][0]) == (i+1):
                temp_npy2 = np.load(paths[j])
                print(paths[j])
        final_npy = np.zeros((1, (temp_npy1.shape[1] + temp_npy2.shape[1])))
        for k in range(l):
            temp_npy = np.hstack((temp_npy1[k,...], temp_npy2[k,...]))
            final_npy = np.vstack((final_npy, temp_npy))
        final_npy = np.delete(final_npy, 0, axis = 0)
        dir_list = [outdir + '/' + 'main', outdir + '/' + 'train', outdir + '/' + 'val' ]
        for dire in dir_list:
            if not os.path.exists(dire):
                os.makedirs(dire)
        np.save(dir_list[0] + '/' + str(i) + '_' + str(i+1) + '_ele_predata', final_npy)
        split_idx = int(final_npy.shape[0] * 0.8)
        np.random.shuffle(final_npy)
        train_data_npy = final_npy[:split_idx, ...]
        np.save(dir_list[1] + '/' + str(i) + '_' + str(i+1) + '_train', train_data_npy)
        val_data_npy = final_npy[split_idx:, ...]
        np.save(dir_list[2] + '/' + str(i) + '_' + str(i+1) + '_val', val_data_npy)


def batch_input(indir, batch_size = 10):
    paths = get_path(indir)
    x = []
    y = []
    sum_len = 0
    corrs_dict = {4:'8', 5:'9', 6:'11', 7:'12', 8:'14', 9:'15'}
    for path in paths:
        print(path)
        num = path.split('/')[-1][0]
        print(num)
        print(type(num))
        print(corrs_dict[int(num)])
        width = int(corrs_dict[int(num)])
        data = np.load(path)
        len = int(data.shape[0]/float(batch_size))
        sum_len += len
        for i in range(len):
            batch_data = data[i * batch_size:(i+1) * batch_size, ...]
            row_len = batch_data.shape[1]
            item_x = torch.from_numpy(batch_data[..., :width])
            item_x = item_x.type(torch.float32)
            item_y = torch.from_numpy(batch_data[..., width:])
            item_y = item_y.type(torch.float32)
            x.append(item_x)
            y.append(item_y)

    return sum_len, x, y


def _train(epoch_idx, num_epoch, model, train_data_dir, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    train_loop = tqdm(range(batch_input(train_data_dir)[0]), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    train_loop.set_description('Epoch {}/{}'.format(epoch_idx + 1, num_epoch))
    x, y = batch_input(train_data_dir)[1:]
    print(len(x))

    for i in train_loop:
        optimizer.zero_grad()
        pred = model(x[i], y[i])
        loss = criterion(pred, y[i])
        epoch_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loop.set_postfix(loss = epoch_loss / (i + 1))

    return epoch_loss / (batch_input(train_data_dir)[0])


def train(train_indir, outdir):
    device = torch.device('cpu')
    model = Transformer().to(device)

    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.L1Loss()

    epoch_num = 200
    clip = 1
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for epoch_idx in range(epoch_num):
        train_loss = _train(epoch_idx, epoch_num, model, train_indir, optimizer, criterion, clip)
        #val_loss
        print('train_loss:', train_loss)
        model_path = outdir + '/' + 'model_save_epoch_{}.pth'.format(epoch_idx)
        #output_dir = 'model_save_epoch_{}'.format(epoch_idx)
        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)
        #print('Saving model to %s' % output_dir)
        #model_to_save = model.module if hasattr(model, 'module') else model
        #result = model_to_save.state_dict().
        torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    #get_data(sys'argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    #get_data_2(sys.argv[1], sys.argv[2])
    train('Linked_Datasets/train', 'model_state_dict')
