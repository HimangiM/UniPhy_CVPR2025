from collections import OrderedDict
import numpy as np
import hydra
import omegaconf
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
from einops.layers.torch import Rearrange
import sys
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle as pkl
import time
import random
import sys, ipdb, traceback, subprocess

torch.autograd.set_detect_anomaly(True)

class FprojNN_StressNN(torch.nn.Module):
    def __init__(self, activation, Ftransform, hidden_size, embed_dim):
        super(FprojNN_StressNN, self).__init__()
        hidden_size = hidden_size

        print ("NN Activation: ", activation)
        if activation == "gelu":
            self.activation = torch.nn.GELU()

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=3, d2=3)

        self.stress_model = nn.Sequential(OrderedDict([
                                          ('fc1_stress', nn.Linear(16+9+embed_dim, hidden_size, bias=True)),
                                          ('act1_stress', self.activation),
                                          ('fc2_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act2_stress', self.activation),
                                          ('fc3_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act3_stress', self.activation),
                                          ('fc4_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act4_stress', self.activation),
                                          ('fc5_stress', nn.Linear(hidden_size, 9, bias=True))
                                          ]))

        self.fproj_model = nn.Sequential(OrderedDict([
                                          ('fc1_fproj', nn.Linear(27+3+embed_dim, hidden_size, bias=True)),
                                          ('act1_fproj', self.activation),
                                          ('fc2_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act2_fproj', self.activation),
                                          ('fc3_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act3_fproj', self.activation),
                                          ('fc4_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act4_fproj', self.activation),
                                          ('fc5_fproj', nn.Linear(hidden_size, 9, bias=True)),
                                          ]))

    def Ftmp_U_sigma_Vt_transform(self, Ftmp):
        U, sigma, Vt = torch.linalg.svd(Ftmp)

        U_flatten = self.flatten(U)  # P x 9
        Vt_flatten = self.flatten(Vt)  # P x 9
        Ftmp_flatten = self.flatten(Ftmp)  # P x 9
        Ftmp_input = torch.cat([Ftmp_flatten, U_flatten, sigma, Vt_flatten], dim=-1)  # P x 27

        return Ftmp_input

    def FFt_logJ_sigma_J_logJ1_J1_transform(self, F):
        Ft = F.transpose(1, 2)    # B x 3 x 3
        FtF = torch.matmul(Ft, F) # B x 3 x 3

        J = torch.max(torch.det(F[:, :, :]), torch.Tensor([1e-6]).cuda())
        J1 = torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda())

        U, sigma, Vt = torch.svd(F)  # sigma = B x 3
        FtF_flatten = self.flatten(FtF)  # P x 9
        J = J.unsqueeze(-1)  # P x 1
        J1 = J1.unsqueeze(-1)  # P x 1

        R = torch.matmul(U, Vt)
        strain = torch.cat([sigma, FtF_flatten, J, torch.log(J), J1, torch.log(J1)], dim=-1)  # P x 16

        return strain, R

    def forward(self, Ftmp, F, C, latent, traj_ids):
        latent_particles = latent(traj_ids)

        Ftmp_flatten = self.Ftmp_U_sigma_Vt_transform(Ftmp)
        out_fproj = self.fproj_model(torch.cat([Ftmp_flatten, latent_particles], dim=-1))
        Fproj = Ftmp + out_fproj.view(out_fproj.shape[0], 3, 3)

        # Stress prediction
        strain, R = self.FFt_logJ_sigma_J_logJ1_J1_transform(F)
        C_flatten = self.flatten(C)
        out_stress = self.stress_model(torch.cat([strain, C_flatten, latent_particles], dim=-1))
        stress = out_stress.view(out_stress.shape[0], 3, 3)
        stress_symmetric = 0.5 * (stress + stress.permute(0, 2, 1))
        stress_symmetric = torch.matmul(R, stress_symmetric)

        return Fproj, stress_symmetric

class F_dataset(torch.utils.data.Dataset):
    def __init__(self, file_dir_list, local_dir):
        self.file_dir_list = file_dir_list
        self.local_dir = local_dir
        self.load_data()

    def load_data(self):
        self.all_data_dict = {}
        self.traj_id_dict = {}

        start_time = time.time()
        for idx, file_dir in enumerate(self.file_dir_list):
            print ("Load data: ", idx)

            idx_l = list(np.arange(0, 500))
            random.shuffle(idx_l)

            cur_Ftmp = torch.load(os.path.join(f"{self.local_dir}/dataset", f"{file_dir}", "GtFtmp.pt"), map_location='cpu')[1:]
            cur_F = torch.load(os.path.join(f"{self.local_dir}/dataset", f"{file_dir}" ,"GtF.pt"), map_location='cpu')[1:]
            cur_stress = torch.load(os.path.join(f"{self.local_dir}/dataset", f"{file_dir}", "GtStress.pt"), map_location='cpu')[1:]
            cur_C = torch.load(os.path.join(f"{self.local_dir}/dataset", f"{file_dir}", "GtC.pt"), map_location='cpu')[1:]
            if cur_C.shape[0] == 960:
                cur_Ftmp = torch.cat((cur_Ftmp, cur_Ftmp, cur_Ftmp[:80, :, :, :]), dim=0)
                cur_F = torch.cat((cur_F, cur_F, cur_F[:80, :, :, :]), dim=0)
                cur_stress = torch.cat((cur_stress, cur_stress, cur_stress[:80, :, :, :]), dim=0)
                cur_C = torch.cat((cur_C, cur_C, cur_C[:80, :, :, :]), dim=0)
            cur_traj_id = torch.full((cur_F.shape[0], cur_F.shape[1]), idx)

            self.sim_timesteps = cur_Ftmp.shape[0]
            self.sim_num_particles = cur_Ftmp.shape[1]

            input_Ftmp_tensor = cur_Ftmp.reshape(-1, 3, 3)
            input_F_tensor = cur_F.reshape(-1, 3, 3)
            gt_stress_tensor = cur_stress.reshape(-1, 3, 3)
            input_C_tensor = cur_C.reshape(-1, 3, 3)
            traj_ids = cur_traj_id.reshape(-1)
            self.traj_id_dict[idx] = f"{file_dir}"

            self.all_data_dict[idx] = {'input_F': input_F_tensor, 'gt_stress': gt_stress_tensor, 'traj_ids': traj_ids,
                                       'input_Ftmp': input_Ftmp_tensor, 'input_C': input_C_tensor}

        print ("Loaded data in: ", time.time() - start_time, " seconds.")

    def __len__(self):
        return len(self.file_dir_list) * self.sim_num_particles * self.sim_timesteps

    def __getitem__(self, idx):
        traj_idx = idx // (self.sim_num_particles * self.sim_timesteps)
        sample_idx = idx % (self.sim_num_particles * self.sim_timesteps)

        sample = {'input_F': self.all_data_dict[traj_idx]['input_F'][sample_idx],
                  'stress_target': self.all_data_dict[traj_idx]['gt_stress'][sample_idx],
                  'input_C': self.all_data_dict[traj_idx]['input_C'][sample_idx],
                  'traj_ids': self.all_data_dict[traj_idx]['traj_ids'][sample_idx],
                  'input_Ftmp': self.all_data_dict[traj_idx]['input_Ftmp'][sample_idx],
                  }
        return sample

@hydra.main(config_path='configs', config_name='default')
def main(cfg: omegaconf.DictConfig):

    ##### Environment and Logging Setup #####
    save_dir = cfg['train_cfg']['save_dir']
    local_dir = cfg['train_cfg']['local_dir']
    logger = logging.getLogger()

    os.makedirs(f"{local_dir}/{save_dir}/logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"{local_dir}/{save_dir}/logs")
    fh = logging.FileHandler(f"{local_dir}/{save_dir}/log.txt")
    with open(f"{local_dir}/{save_dir}/config.yaml", 'w') as cfg_file:
        OmegaConf.save(cfg, cfg_file)

    fh.setLevel(logging.DEBUG) # or any level you want
    logger.addHandler(fh)

    ##### Load Dataset #####
    traj_l = ["elastic_diverse", "newtonian_diverse", "non_newtonian_diverse", "plasticine_diverse", "sand_diverse"]
    traj_dir_list = []
    for tl in traj_l:
        for _dir_idx, _dir in enumerate(os.listdir(os.path.join(f"{local_dir}/dataset/", tl))):
            traj_dir_list.append(f"{tl}/{_dir}")

    count_traj = len(traj_dir_list)
    train_data = F_dataset(file_dir_list=traj_dir_list, local_dir=local_dir)
    train_loader = DataLoader(train_data, batch_size=cfg['train_cfg']['batch_size'], shuffle=False, drop_last=False)
    print ("Dataset size: ", len(train_data), "Number of trajectories: ", count_traj)
    with open(os.path.join(local_dir, save_dir, f"traj_name_id_{count_traj}.pkl"), "wb") as f:
        pkl.dump(train_data.traj_id_dict, f)

    ##### Model Setup #####

    # Define Latent
    embed_dim = cfg['train_cfg']['embed_dim']
    trajectory_latent = torch.nn.Embedding(count_traj, embed_dim).cuda()
    traj_mean, traj_std = 0., 1.
    torch.nn.init.normal_(trajectory_latent.weight, mean=traj_mean, std=traj_std)

    # Define Model
    model = FprojNN_StressNN(activation=cfg['train_cfg']['nn_activation'],
                            Ftransform=cfg['train_cfg']['Ftransform'],
                            hidden_size=cfg['train_cfg']['hidden_size'],
                            embed_dim=cfg['train_cfg']['embed_dim']).cuda()
    model.train()
    
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": cfg['train_cfg']['lr']},
                                   {"params": trajectory_latent.parameters(), "lr": cfg['train_cfg']['lr'] * 10.}])
    total_epochs = 2000000
    step_lr_step_size=cfg['train_cfg']['step_lr_step_size']
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_lr_step_size, gamma=0.9)

    loss_mse = nn.MSELoss()
    loss_mse_fproj = nn.MSELoss()
    for epoch in range(total_epochs):
        print ("Epoch: ", epoch)
        for idx, data in enumerate(train_loader):
            input_F, stress_target, traj_ids, input_Ftmp, input_C = data['input_F'].cuda(), data['stress_target'].cuda(), data['traj_ids'].cuda(), data['input_Ftmp'].cuda(), data['input_C'].cuda()

            optimizer.zero_grad()

            pred_Fproj, pred_stress = model(input_Ftmp, input_F, input_C, trajectory_latent, traj_ids)

            if len((torch.isnan(pred_stress) == True).nonzero()) > 0 or len((torch.isnan(pred_Fproj) == True).nonzero()) > 0:
                import ipdb; ipdb.set_trace()

            loss_l2 = loss_mse(pred_stress, stress_target)
            loss_l2_fproj = loss_mse_fproj(pred_Fproj, input_F)

            reg_loss = torch.norm(trajectory_latent.weight, dim=-1)
            loss_reg = 0.0001 * torch.mean(reg_loss)
            loss = loss_l2 + loss_l2_fproj + loss_reg

            loss.backward()
            optimizer.step()

            logger.info(f"Epoch: {epoch}, Iter/Total: {idx}/{len(train_loader)}, Loss: {loss}, LR1: {scheduler1.get_last_lr()}")
            if idx % 1000 == 0:
                writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + idx)
                writer.add_scalar('Stress Loss/train', loss_l2, epoch * len(train_loader) + idx)
                writer.add_scalar('Fproj Loss/train', loss_l2_fproj, epoch * len(train_loader) + idx)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + idx)

            if epoch < 2:
                scheduler1.step()

        if epoch % 1 == 0:
            torch.save(trajectory_latent, f"{local_dir}/{save_dir}/traj_latent_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f"{local_dir}/{save_dir}/model_{epoch}.pth")

        train_loader.dataset.load_data()

if __name__=='__main__':
    main()








