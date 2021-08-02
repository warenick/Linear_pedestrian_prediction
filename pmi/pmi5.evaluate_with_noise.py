import sys
sys.path.append("./../")
import torch
from LPP import LPP
from pedestrian_forecasting_dataloader.config import cfg
from pedestrian_forecasting_dataloader.train_test_split import get_dataloaders
from pedestrian_forecasting_dataloader.DataNoiser import DataNoiser
from tqdm import tqdm
import numpy as np
import argparse
torch.set_printoptions(precision=1)
cfg["raster_params"]["draw_hist"] = False
cfg["raster_params"]["use_map"] = False
cfg["raster_params"]["normalize"] = True
cfg["one_ped_one_traj"] = False

def get_neighb_poses(data):
    num_peds = 0
    for sequence in data.history_agents:
        num_peds = max(num_peds, len(sequence))
    neighb_poses = -1 * torch.ones((len(data.history_agents), num_peds, 8, 6))
    neighb_poses_avail = torch.zeros((len(data.history_agents), num_peds, 8))
    for i in range(len(data.history_agents)):
        for j in range(len(data.history_agents[i])):
            try:
                neighb_poses[i, j] = torch.tensor(data.history_agents[i][j])
                neighb_poses_avail[i, j] = torch.tensor(data.history_agents_avail[i][j])
            except:
                break
    neighb_poses = neighb_poses.float()
    return neighb_poses, neighb_poses_avail


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


def eval_data(dataloader, lpp, data_noiser = None, noise_model = None,dev='cpu'):
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
        if data_noiser is not None:
            neighb_poses, neighb_poses_avail = get_neighb_poses(data)
            neighb_poses = neighb_poses[b_mask]
            neighb_poses_avail = neighb_poses_avail[b_mask]
            self_poses_av = torch.ones(self_poses.shape[:self_poses.dim()-1])
            neighb_poses_avail = torch.ones(neighb_poses.shape[:neighb_poses.dim()-1])
            if noise_model["pose_noise"] >0.:
                self_poses = data_noiser.pose_noise(self_poses,sigma=noise_model["pose_noise"])
                neighb_poses = data_noiser.pose_noise(neighb_poses,sigma=noise_model["pose_noise"]) #TODO: check available
            if noise_model["id_noise"]>0:
                self_poses, self_poses_av, _, _ = data_noiser.id_batch_noise(self_poses, self_poses_av, neighb_poses, neighb_poses_avail,num_steps=noise_model["id_noise"])
        mean_poses = lpp.predict(
                state = self_poses[:, 0, :2],
                # goal= gt_goals)
                history= self_poses[:, 1:, :2],
                avail = self_poses_av)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_noise', type=str, default="None")
    parser.add_argument('--segm_noise', type=str, default="None")
    parser.add_argument('--pose_noise', type=float, default=0.)
    parser.add_argument("--id_noise", type=int, default=0)
    parser.add_argument("--validate_with", type=str, default="SDD")
    args = parser.parse_args()
    noise_model = {"rgb_noise":args.rgb_noise,
                    "segm_noise":args.segm_noise,
                    "pose_noise":args.pose_noise,
                    "id_noise":args.id_noise
                    }
    noise_model["id_noise"]=1
    dn = DataNoiser(seed=7)
    dev = 'cpu'
    lpp = LPP()
    ades = []
    fdes = []
    path_ = "../pedestrian_forecasting_dataloader/data/train"
    # validation = ['eth_hotel', 'biwi_eth', 'zara01', 'zara02', 'students', 'SDD']
    validation = [args.validate_with]
    # validation = ['eth_hotel']
    # validation = ['SDD']
    print(noise_model)
    for validate_with in validation:
        with torch.no_grad():
            _, val_dataloader = get_dataloaders(bs=256, num_w=0, path_=path_, validate_with=validate_with, cfg_=cfg)
            ade, fde = eval_data(val_dataloader,lpp,dn,noise_model)
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

# best
# velocity = torch.mean(history[:,0:-1]-history[:,1:],dim=1)
# eth_hotel  ade  0.29 	 fde  0.58
# biwi_eth 	 ade  0.63 	 fde  1.28
# zara01 	 ade  0.63 	 fde  1.29
# zara02 	 ade  0.47 	 fde  0.96
# students 	 ade  0.74 	 fde  1.50
# SDD 	     ade  0.74 	 fde  1.46

# Nose model experements:
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 0}
# dataset  eth_hotel 	 ade  0.297 	 fde  0.583
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.01, 'id_noise': 0}
# dataset  eth_hotel 	 ade  0.304 	 fde  0.592
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.05, 'id_noise': 0}
# dataset  eth_hotel 	 ade  0.347 	 fde  0.656
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.1, 'id_noise': 0}
# dataset  eth_hotel 	 ade  0.425 	 fde  0.773
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.2, 'id_noise': 0}
# dataset  eth_hotel 	 ade  0.616 	 fde  1.06
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.4, 'id_noise': 0}
# dataset  eth_hotel 	 ade  1.05 	 fde  1.72
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.8, 'id_noise': 0}
# dataset  eth_hotel 	 ade  1.98 	 fde  3.17
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 1.5, 'id_noise': 0}
# dataset  eth_hotel 	 ade  3.65 	 fde  5.78
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 1}
# dataset  eth_hotel 	 ade  64.1 	 fde  1.06e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 2}
# dataset  eth_hotel 	 ade  82.4 	 fde  1.38e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 3}
# dataset  eth_hotel 	 ade  62.7 	 fde  98.1
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 4}
# dataset  eth_hotel 	 ade  68.6 	 fde  1.02e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 0}
# dataset  biwi_eth 	 ade  0.637 	 fde  1.28
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.01, 'id_noise': 0}
# dataset  biwi_eth 	 ade  0.638 	 fde  1.28
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.05, 'id_noise': 0}
# dataset  biwi_eth 	 ade  0.654 	 fde  1.3
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.1, 'id_noise': 0}
# dataset  biwi_eth 	 ade  0.694 	 fde  1.35
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.2, 'id_noise': 0}
# dataset  biwi_eth 	 ade  0.824 	 fde  1.52
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.4, 'id_noise': 0}
# dataset  biwi_eth 	 ade  1.19 	 fde  2.03
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.8, 'id_noise': 0}
# dataset  biwi_eth 	 ade  2.05 	 fde  3.33
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 1.5, 'id_noise': 0}
# dataset  biwi_eth 	 ade  3.68 	 fde  5.84
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 1}
# dataset  biwi_eth 	 ade  75.6 	 fde  1.27e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 2}
# dataset  biwi_eth 	 ade  75.0 	 fde  1.22e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 3}
# dataset  biwi_eth 	 ade  76.6 	 fde  1.25e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 4}
# dataset  biwi_eth 	 ade  95.5 	 fde  1.54e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 0}
# dataset  zara01 	 ade  0.634 	 fde  1.29
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.01, 'id_noise': 0}
# dataset  zara01 	 ade  0.635 	 fde  1.29
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.05, 'id_noise': 0}
# dataset  zara01 	 ade  0.653 	 fde  1.31
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.1, 'id_noise': 0}
# dataset  zara01 	 ade  0.698 	 fde  1.37
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.2, 'id_noise': 0}
# dataset  zara01 	 ade  0.838 	 fde  1.56
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.4, 'id_noise': 0}
# dataset  zara01 	 ade  1.21 	 fde  2.1
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.8, 'id_noise': 0}
# dataset  zara01 	 ade  2.08 	 fde  3.41
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 1.5, 'id_noise': 0}
# dataset  zara01 	 ade  3.7 	 fde  5.92
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 1}
# dataset  zara01 	 ade  70.8 	 fde  1.11e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 2}
# dataset  zara01 	 ade  66.9 	 fde  1.04e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 3}
# dataset  zara01 	 ade  81.5 	 fde  1.3e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 4}
# dataset  zara01 	 ade  72.3 	 fde  1.04e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 0}
# dataset  zara02 	 ade  0.478 	 fde  0.967
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.01, 'id_noise': 0}
# dataset  zara02 	 ade  0.484 	 fde  0.977
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.05, 'id_noise': 0}
# dataset  zara02 	 ade  0.53 	 fde  1.04
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.1, 'id_noise': 0}
# dataset  zara02 	 ade  0.602 	 fde  1.15
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.2, 'id_noise': 0}
# dataset  zara02 	 ade  0.773 	 fde  1.4
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.4, 'id_noise': 0}
# dataset  zara02 	 ade  1.17 	 fde  1.99
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.8, 'id_noise': 0}
# dataset  zara02 	 ade  2.05 	 fde  3.33
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 1.5, 'id_noise': 0}
# dataset  zara02 	 ade  3.67 	 fde  5.86
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 1}
# dataset  zara02 	 ade  60.4 	 fde  91.7
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 2}
# dataset  zara02 	 ade  50.3 	 fde  83.9
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 3}
# dataset  zara02 	 ade  60.0 	 fde  96.2
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 4}
# dataset  zara02 	 ade  57.4 	 fde  89.7
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 0}
# dataset  students 	 ade  0.749 	 fde  1.5
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.01, 'id_noise': 0}
# dataset  students 	 ade  0.751 	 fde  1.5
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.05, 'id_noise': 0}
# dataset  students 	 ade  0.771 	 fde  1.53
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.1, 'id_noise': 0}
# dataset  students 	 ade  0.813 	 fde  1.58
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.2, 'id_noise': 0}
# dataset  students 	 ade  0.935 	 fde  1.74
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.4, 'id_noise': 0}
# dataset  students 	 ade  1.27 	 fde  2.21
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.8, 'id_noise': 0}
# dataset  students 	 ade  2.1 	 fde  3.44
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 1.5, 'id_noise': 0}
# dataset  students 	 ade  3.68 	 fde  5.9
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 1}
# dataset  students 	 ade  57.2 	 fde  89.3
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 2}
# dataset  students 	 ade  59.3 	 fde  94.1
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 3}
# dataset  students 	 ade  57.3 	 fde  90.8
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 4}
# dataset  students 	 ade  59.5 	 fde  94.4
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 0}
# dataset  SDD 	 ade  0.747 	 fde  1.46
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.01, 'id_noise': 0}
# dataset  SDD 	 ade  0.749 	 fde  1.46
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.05, 'id_noise': 0}
# dataset  SDD 	 ade  0.776 	 fde  1.5
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.1, 'id_noise': 0}
# dataset  SDD 	 ade  0.829 	 fde  1.57
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.2, 'id_noise': 0}
# dataset  SDD 	 ade  0.969 	 fde  1.77
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.4, 'id_noise': 0}
# dataset  SDD 	 ade  1.32 	 fde  2.29
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.8, 'id_noise': 0}
# dataset  SDD 	 ade  2.15 	 fde  3.55
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 1.5, 'id_noise': 0}
# dataset  SDD 	 ade  3.72 	 fde  5.98
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 1}
# dataset  SDD 	 ade  85.1 	 fde  1.31e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 2}
# dataset  SDD 	 ade  93.7 	 fde  1.45e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 3}
# dataset  SDD 	 ade  92.0 	 fde  1.44e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 4}
# dataset  SDD 	 ade  90.0 	 fde  1.43e+02














# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 0}
# dataset  SDD 	 ade  0.747 	 fde  1.46

# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.01, 'id_noise': 0}
# dataset  SDD 	 ade  0.749 	 fde  1.46
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.05, 'id_noise': 0}
# dataset  SDD 	 ade  0.776 	 fde  1.5
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.1, 'id_noise': 0}
# dataset  SDD 	 ade  0.829 	 fde  1.57
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.2, 'id_noise': 0}
# dataset  SDD 	 ade  0.969 	 fde  1.77

# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 1}
# dataset  SDD 	 ade  96.3 	 fde  1.48e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 2}
# dataset  SDD 	 ade  97.0 	 fde  1.55e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 3}
# dataset  SDD 	 ade  90.4 	 fde  1.37e+02
# {'rgb_noise': 'None', 'segm_noise': 'None', 'pose_noise': 0.0, 'id_noise': 4}
# dataset  SDD 	 ade  92.7 	 fde  1.37e+02
