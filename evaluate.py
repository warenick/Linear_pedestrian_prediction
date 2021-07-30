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
from gp_no_map import AttGoalPredictor, LSTM_simple

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


def eval_file(path_,file,lpp,gp_model,dev='cpu'):
    print("used files: ",file[0])
    dataset = DatasetFromTxt(path_, file, cfg)
    print("len dataset:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_wrapper, pin_memory=True)

    pbar = tqdm(dataloader)
    ades = []
    fdes = []
    dists = []
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

# gp
        if gp_model is not None:
            num_peds = 0 #
            i=0 #
            for sequence in data.history_agents:
                if (b_mask[i]): #
                    num_peds = max(num_peds, len(sequence))
                i=i+1 #

            neighb_poses = torch.zeros((bs, num_peds, 8, 6),device=dev)
            neighb_poses_avail = torch.zeros((bs, num_peds, 8),device=dev)
            for i in range(len(data.history_agents)):
                if (b_mask[i]):
                    for j in range(num_peds):
                        try:
                            neighb_poses[i, j] = torch.tensor(data.history_agents[i][j],device=dev,dtype=torch.float)
                            neighb_poses_avail[i, j] = torch.tensor(data.history_agents_avail[i][j],device=dev,dtype=torch.float)
                        except:
                            # ????
                            pass
            predictions = gp_model(self_poses, neighb_poses)
            mean_poses = lpp.predict(
                state = self_poses[:, 0, :2],
                goal= predictions)
# gp
        else:
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
        d = torch.tensor(0)
        if gp_model is not None:
            d = distance(predictions, gt_goals, mask_goal)
        dists.append(d)
        pbar.set_postfix({' bs ': bs})
        # print('bs', bs)
    pbar.close()
    try:
        ade = sum(ades)/len(ades)
        fde = (sum(fdes)/len(fdes)).tolist()
        dist = (sum(dists)/len(dists)).tolist()
    except:
        print("dividing by zero")
        ade = 10
        fde = 10
        dist = 10

    return ade, fde, dist 



if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = torch.device(0)
    dev = 'cpu'
    # init LPP
    lpp = LPP()
    # gp prediction
    gp_model = LSTM_simple()
    gp_model.eval()
    gp_model = gp_model.to(dev)
    gp_model_path = "gp_model.pth"
    checkpoint = torch.load(gp_model_path)
    gp_model.load_state_dict(checkpoint['model_state_dict'])
    # gp_model = None
    # path_ = "pedestrian_forecasting_dataloader/data/test/"
    path_ = "pedestrian_forecasting_dataloader/data/train/ros/"
    # get all availible data
    pathes = list(Path(path_).rglob("*.[tT][xX][tT]"))
    files = [str(x).replace(path_,"") for x in pathes]
    # files = ["output.txt"]
    # files = ['biwi_eth/biwi_eth.txt']
    ades = []
    fdes = []
    with torch.no_grad():
        for file in files:
            ade, fde, dist = eval_file(path_,[file],lpp,gp_model,dev)
            ades.append(ade)
            fdes.append(fde)
            print("\nfile ",file,"\n\t ade ",ade,"\t fde ",fde,"\t dist ",dist)
    ades = sum(ades)/len(ades)
    fdes = sum(fdes)/len(fdes)
    print("\n\t ade ",ades,"\t fde ",fdes)

    with open("optimise_data.txt","a") as sfile:
        sfile.write("\n\t ade "+ str(ades)+"\t fde "+ str(fdes))
        
    exit()
# velocity = torch.mean(history[:,0:-1]-history[:,1:],dim=1) 
# ade  1.7082285225331095         fde  3.562949786228793 
# velocity = (history[:,0]-history[:,-1])/len(history[0])
# ade  1.666416962826211          fde  3.400552095047065
# velocity = history[:,0]-history[:,1] #
# ade  1.4707280321023115         fde  3.183572230594499
# +GP
# ade  1.5556726177717048         fde  3.005297842834677