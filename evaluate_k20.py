import sys
import os
import torch
from LPP import LPP
from pathlib import Path
from pedestrian_forecasting_dataloader.dataloader import DatasetFromTxt, collate_wrapper
from pedestrian_forecasting_dataloader.config import cfg
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

cfg["raster_params"]["draw_hist"] = False
cfg["raster_params"]["use_map"] = False
cfg["raster_params"]["normalize"] = True

def ade_loss(pred_traj, gt, mask):
    assert pred_traj.ndim == 3
    assert gt.ndim == 3
    assert mask.ndim == 2
    error = pred_traj - gt
    norm = torch.norm(error, dim=2)[mask]
    return torch.mean(norm)


def distance(prediction, gt, tgt_avail=None):
    if prediction.ndim == 3:
        return oracle_distance(prediction, gt, tgt_avail)
    if tgt_avail is None:
        return torch.mean(torch.sqrt(torch.sum((prediction - gt) ** 2, dim=1)))
    tgt_avail = tgt_avail.bool().to(prediction.device)
    error = prediction - gt
    norm = torch.norm(error, dim=1)
    error_masked = norm[tgt_avail]
    if torch.sum(tgt_avail) != 0:
        return torch.mean(error_masked)
    else:
        print("no gt available?")
        return 0

def generate_poses(state,rad,k):
    out = []
    out.append(state.clone())
    for i in range(k-1):
        sample = state.clone()
        sample[:,0]+=(torch.rand([state.shape[0]])*2-1.)*rad
        sample[:,1]+=(torch.rand([state.shape[0]])*2-1.)*rad
        out.append(sample)
    return out

def eval_file(path_,file,lpp,dev='cpu',k=20):
    print("used files: ",file[0])
    dataset = DatasetFromTxt(path_, file, cfg)
    print("len dataset:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_wrapper, pin_memory=True)

    pbar = tqdm(dataloader)
    ades = []
    fdes = []
    for batch_num, data in enumerate(pbar):

        if np.sum(data.tgt_avail[:, -1]) == 0:
            # print("no one have goal")
            continue
        b_mask = (np.sum(data.tgt_avail,axis=1)==12)*(np.sum(data.history_av,axis=1)==8)
        self_poses = torch.tensor(data.history_positions,device=dev,dtype=torch.float)[b_mask] #
        bs = self_poses.shape[0]
        if bs<1:
            # print("no one have full history batch")
            continue
        gt_goals = torch.tensor(data.tgt[:, -1, :],device=dev,dtype=torch.float)[b_mask] #
        traj_tgt = torch.tensor(data.tgt,device=dev,dtype=torch.float)[b_mask] #
        mask_goal = torch.tensor(data.tgt_avail[:, -1],device=dev,dtype=torch.bool)[b_mask] #
        mask_traj = torch.tensor(data.tgt_avail,device=dev,dtype=torch.bool)[b_mask] #
        mp = []
        l2s = []
        pose_sample = generate_poses(self_poses[:, 0, :2],rad=1.0,k=20)
        for sample in pose_sample:
            mean_poses = lpp.predict(
                state = sample,
                history= self_poses[:, 1:, :2])
            mp.append(mean_poses)
            l2s.append(torch.norm(mean_poses[:,-1,:2]-gt_goals))
        l2s = torch.stack(l2s)
        ids = torch.argmin(l2s)
        ades.append(ade_loss(mp[ids][:, :, :2].detach(), traj_tgt, mask_traj).item())
        fdes.append(distance(mp[ids][:,-1,:2], gt_goals, mask_goal))
        pbar.set_postfix({' bs ': bs})
    pbar.close()
    try:
        ade = sum(ades)/len(ades)
        fde = (sum(fdes)/len(fdes)).tolist()
    except:
        print("dividing by zero")
        ade = 10
        fde = 10

    return ade, fde



if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = torch.device(0)
    dev = 'cpu'
    # init LPP
    lpp = LPP()
    path_ = "pedestrian_forecasting_dataloader/data/temp/"
    # get all availible data
    pathes = list(Path(path_).rglob("*.[tT][xX][tT]"))
    files = [str(x).replace(path_,"") for x in pathes]
    # files = ['biwi_eth/biwi_eth.txt']
    ades = []
    fdes = []
    with torch.no_grad():
        for file in files:
            ade, fde = eval_file(path_,[file],lpp,dev,k=20)
            ades.append(ade)
            fdes.append(fde)
            print("\nfile ",file,"\n\t ade ",ade,"\t fde ",fde)
    ades = sum(ades)/len(ades)
    fdes = sum(fdes)/len(fdes)
    print("\n\t ade ",ades,"\t fde ",fdes)

    exit()
# on sdd