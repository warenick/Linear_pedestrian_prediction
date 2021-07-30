import sys
sys.path.append("./../")
import torch
from LPP import LPP
from pedestrian_forecasting_dataloader.config import cfg
from pedestrian_forecasting_dataloader.train_test_split import get_dataloaders
from tqdm import tqdm
import numpy as np

cfg["raster_params"]["draw_hist"] = False
cfg["raster_params"]["use_map"] = False
cfg["raster_params"]["normalize"] = True
cfg["one_ped_one_traj"] = False


def ade_loss(pred_traj, gt, mask):
    assert pred_traj.ndim == 3
    assert gt.ndim == 3
    assert mask.ndim == 2
    error = pred_traj - gt
    norm = torch.norm(error, dim=2)[mask]
    return torch.mean(norm)


def distance(prediction, gt, tgt_avail=None):
    if prediction.ndim == 3:
        return oracle_distance(prediction, gt, tgt_avail)# newer comes here)) TODO: realize that function once it would happend
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
        return torch.tensor(0)


def eval_data(dataloader,lpp,dev='cpu'):
    pbar = tqdm(dataloader)
    ades = []
    fdes = []
    for batch_num, data in enumerate(pbar):

        # if np.sum(data.tgt_avail[:, -1]) == 0:
        #     print("no one have goal")
        #     continue
        b_mask = (np.sum(data.tgt_avail,axis=1)==12)*(np.sum(data.history_av,axis=1)==8)
        self_poses = torch.tensor(data.history_positions,device=dev,dtype=torch.float)[b_mask] #
        bs = self_poses.shape[0]
        if bs<1:
            print("no one have full history batch")
            continue
        gt_goals = torch.tensor(data.tgt[:, -1, :],device=dev,dtype=torch.float)[b_mask] #
        traj_tgt = torch.tensor(data.tgt,device=dev,dtype=torch.float)[b_mask] #
        mean_poses = lpp.predict(
                state = self_poses[:, 0, :2],
                # goal= gt_goals)
                history= self_poses[:, 1:, :2])
        if len(mean_poses[mean_poses != mean_poses]) != 0:
            print("nans!")
            continue
        mask_goal = torch.tensor(data.tgt_avail[:, -1],device=dev,dtype=torch.bool)[b_mask] #
        mask_traj = torch.tensor(data.tgt_avail,device=dev,dtype=torch.bool)[b_mask] #
        ades.append(ade_loss(mean_poses[:, :, :2].detach(), traj_tgt, mask_traj).item())
        fdes.append(distance(mean_poses[:,-1,:2], gt_goals, mask_goal))
        pbar.set_postfix({' bs ': bs})
        # print('bs', bs)
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
    dev = 'cpu'
    lpp = LPP()
    ades = []
    fdes = []
    path_ = "../pedestrian_forecasting_dataloader/data/train"
    validation = ['eth_hotel', 'biwi_eth', 'zara01', 'zara02', 'students', 'SDD']
    for validate_with in validation:
        with torch.no_grad():
            _, val_dataloader = get_dataloaders(bs=256, num_w=0, path_=path_, validate_with=validate_with, cfg_=cfg)
            ade, fde = eval_data(val_dataloader,lpp)
            print("dataset ",validate_with,"\t ade ",f'{ade:.3}',"\t fde ",f'{fde:.3}')
    exit()
# velocity = history[:,0]-history[:,1]
# eth_hotel  ade  0.37 	 fde  0.72
# biwi_eth 	 ade  0.73 	 fde  1.45
# zara01 	 ade  0.54 	 fde  1.15
# zara02 	 ade  0.41 	 fde  0.87
# students 	 ade  0.64 	 fde  1.35
# SDD 	     ade  0.80 	 fde  1.59

# velocity = (history[:,0]-history[:,-1])/len(history[0])
# eth_hotel  ade  0.31 	 fde  0.58
# biwi_eth 	 ade  0.73 	 fde  1.41
# zara01 	 ade  0.73 	 fde  1.44
# zara02 	 ade  0.50 	 fde  0.98
# students 	 ade  0.71 	 fde  1.41
# SDD 	     ade  0.82 	 fde  1.57

# velocity = torch.mean(history[:,0:-1]-history[:,1:],dim=1)
# eth_hotel  ade  0.29 	 fde  0.58
# biwi_eth 	 ade  0.63 	 fde  1.28
# zara01 	 ade  0.63 	 fde  1.29
# zara02 	 ade  0.47 	 fde  0.96
# students 	 ade  0.74 	 fde  1.50
# SDD 	     ade  0.74 	 fde  1.46



