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
from mpmwrapper import MPMWrapperLearnableStress
import taichi as ti
import random

from houdini_visualization_gradio import visualize_simulation

torch.autograd.set_detect_anomaly(True)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the random seed
seed = 42
set_random_seed(seed)

class Latent(torch.nn.Module):
    def __init__(self, device, traj_latent_path, embed_dim):
        super(Latent, self).__init__()
        self.traj_latent_path = traj_latent_path
        self.embed_dim = embed_dim

        traj_path = self.traj_latent_path
        trajectory_latent_embedding_orig = torch.load(traj_path).weight.detach()
        trajectory_latent_embedding = torch.Tensor(trajectory_latent_embedding_orig)
        self.trajectory_latent = torch.nn.Embedding.from_pretrained(trajectory_latent_embedding, freeze=False).to(device)

    def forward(self):
        pass

class FprojNN(torch.nn.Module):
    def __init__(self, activation, hidden_size, device, embed_dim, trajectory_latent):
        super(FprojNN, self).__init__()
        hidden_size = hidden_size
        self.device = device

        if activation == "gelu":
            self.activation = torch.nn.GELU()

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=3, d2=3)
        self.device = "cuda"

        self.fc1 = nn.Linear(27+embed_dim, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, 9, bias=True)

        self.trajectory_latent = trajectory_latent

    def Ftmp_U_Vt_transform(self, Ftmp, U, V):
        if len((torch.isnan(Ftmp) == True).nonzero()) > 0 or len((torch.isinf(Ftmp) == True).nonzero()) > 0:
            import ipdb; ipdb.set_trace()

        U_flatten = self.flatten(U)  # P x 9
        Vt_flatten = self.flatten(V.transpose(1, 2))  # P x 9
        Ftmp_flatten = self.flatten(Ftmp)  # P x 9
        Ftmp_input = torch.cat([Ftmp_flatten, U_flatten, Vt_flatten], dim=-1)  # P x 27

        return Ftmp_input

    def forward(self, Ftmp, U, V, traj_id):
        Ftmp_flatten = self.Ftmp_U_Vt_transform(Ftmp, U, V)

        latent_particles = self.trajectory_latent.weight[traj_id].unsqueeze(0).repeat(Ftmp.shape[0], 1)
        x = self.activation(self.fc1(torch.cat([Ftmp_flatten, latent_particles], dim=-1)))
        x = self.activation(self.fc2(x))
        out = self.fc3(x)

        Fproj = Ftmp + out.view(out.shape[0], 3, 3)

        return Fproj

class StressNN(torch.nn.Module):
    def __init__(self, activation, hidden_size, embed_dim, device, trajectory_latent):
        super(StressNN, self).__init__()
        hidden_size = hidden_size
        self.device = device

        if activation == "gelu":
            self.activation = torch.nn.GELU()

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=3, d2=3)
        self.device = "cuda"

        self.fc1 = nn.Linear(16+9+embed_dim, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc4 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc5 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc6 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc7 = nn.Linear(hidden_size, 9, bias=True)

        self.trajectory_latent = trajectory_latent

    def FFt_logJ_sigma_J_logJ1_J1_transform(self, F):
        Ft = F.transpose(1, 2)    # B x 3 x 3
        FFt = torch.matmul(F, Ft) # B x 3 x 3

        J = torch.max(torch.det(F[:, :, :]), torch.Tensor([1e-6]).cuda())
        J1 = torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda())

        _, sigma, _ = torch.svd(F)  # sigma = B x 3
        FFt_flatten = self.flatten(FFt)  # P x 9
        J = J.unsqueeze(-1)  # P x 1
        J1 = J1.unsqueeze(-1)  # P x 1

        strain = torch.cat([FFt_flatten, torch.log(J), sigma, J, torch.log(J1), J1], dim=-1)  # P x 16

        return strain

    def forward(self, F, C, traj_id):
        strain = self.FFt_logJ_sigma_J_logJ1_J1_transform(F)

        latent_particles = self.trajectory_latent.weight[traj_id].unsqueeze(0).repeat(F.shape[0], 1)
        C_flatten = self.flatten(C)
        x = self.activation(self.fc1(torch.cat([strain, C_flatten, latent_particles], dim=-1)))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.activation(self.fc6(x))
        out = self.fc7(x)

        stress = out.view(out.shape[0], 3, 3)
        stress_symmetric = 0.5 * (stress + stress.permute(0, 2, 1))

        return stress_symmetric

@hydra.main(config_path='configs', config_name='default')
def main(cfg: omegaconf.DictConfig):

    ## Logging ##
    save_dir = cfg['train_cfg']['save_dir']
    local_dir = cfg['train_cfg']['local_dir']
    logger = logging.getLogger()

    device = 'cuda'
    num_sim_steps = cfg['visualization_cfg']['num_frames']
    cuda_chunk_size = cfg['train_cfg']['cuda_chunk_size']
    particles_ti_root = cfg['train_cfg']['particles_ti_root']

    os.makedirs(f"{local_dir}/{save_dir}/logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"{local_dir}/{save_dir}/logs")
    fh = logging.FileHandler(f"{local_dir}/{save_dir}/log.txt")
    with open(f"{local_dir}/{save_dir}/config.yaml", 'w') as cfg_file:
        OmegaConf.save(cfg, cfg_file)

    fh.setLevel(logging.DEBUG) # or any level you want
    logger.addHandler(fh)

    ## Load the model ##
    fproj_model = FprojNN(activation=cfg['train_cfg']['nn_activation'],
                          hidden_size=cfg['train_cfg']['hidden_size'],
                          device='cuda',
                          embed_dim=cfg['train_cfg']['embed_dim'],
                          trajectory_latent=None).to(device)

    stress_model = StressNN(activation=cfg['train_cfg']['nn_activation'],
                            hidden_size=cfg['train_cfg']['hidden_size'],
                            embed_dim=cfg['train_cfg']['embed_dim'],
                            device='cuda',
                            trajectory_latent=None).to(device)

    if cfg['train_cfg']['load_model']:
        ckpt = torch.load(cfg['train_cfg']['load_model'])
        ckpt_stress = {key.replace('module.', '').replace('_stress', '').replace('stress_model.', ''): value for key, value in ckpt.items() if 'stress' in key}
        ckpt_fproj = {key.replace('module.', '').replace('_fproj', '').replace('fproj_model.', ''): value for key, value in ckpt.items() if 'fproj' in key}
        stress_model.load_state_dict(ckpt_stress)
        fproj_model.load_state_dict(ckpt_fproj)

    fproj_model.eval()
    stress_model.eval()

    # Load trajectory data
    traj_data_dir = cfg['train_cfg']['traj_data_dir']
    traj_data_orig = torch.load(os.path.join(traj_data_dir, 'GtX.pt'))
    traj_data_orig = torch.tensor(traj_data_orig).to(device) # T x P x 3

    traj_idx = 0

    ## Load the trajectory latent ##
    latent_obj = Latent(device="cuda", traj_latent_path=cfg['train_cfg']['traj_latent_path'], embed_dim=cfg['train_cfg']['embed_dim'])
    latent_obj.train()

    fproj_model.trajectory_latent = latent_obj.trajectory_latent
    stress_model.trajectory_latent = latent_obj.trajectory_latent

    ## Load the specific trajectory object's config ##
    with open(os.path.join(traj_data_dir, "config.yaml")) as traj_cfg_f:
        traj_cfg = OmegaConf.load(traj_cfg_f)

    ti_mem_fraction = 0.7
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=ti_mem_fraction, debug=True)

    # Load positions
    init_particles_simulator = traj_data_orig[0].contiguous()
    target_particles_simulator = traj_data_orig.permute(1, 0, 2).cpu().numpy()  # P x T x 3

    # Assign the specific trajectory object's config
    cfg['objects'] = traj_cfg['objects']

    # Initialize MPMWrapper and simulator
    mpmwrapper_learnable = MPMWrapperLearnableStress(objects=cfg['objects'], simulator_cfg=cfg['simulator_cfg'],
                                                       visualization_cfg=cfg['visualization_cfg'],
                                                       init_particles=init_particles_simulator, target_particles=target_particles_simulator,
                                                       constitutive_function=None, particles_ti_root=particles_ti_root,
                                                       cuda_chunk_size=cuda_chunk_size, stress_model=stress_model, tb_writer=None,
                                                       stress_data_gt=None, embed_dim=cfg['train_cfg']['embed_dim'],
                                                       fproj_model=fproj_model)

    # Initialize simulator and wrapper variables
    mpmwrapper_learnable.initialize_particles()
    mpmwrapper_learnable.simulator_variables_initialize()

    mpmwrapper_learnable.update_traj_latent_learnable(latent_obj.trajectory_latent.weight[traj_idx])

    substep_pred = mpmwrapper_learnable.simulator.n_substeps[None]

    for i in range(num_sim_steps):
        print(f"Sim step: {i}")

        for step in range(substep_pred * i, substep_pred * (i + 1)):
            local_index = step % cuda_chunk_size

            mpmwrapper_learnable.simulator.advance_F_stress(t=local_index, traj_id=traj_idx)

            if not mpmwrapper_learnable.simulator.cfl_satisfy[None]:
                mpmwrapper_learnable.simulator.cached_states.clear()
                print ("Cfl not satisfied", f"Sim step: {i}, Step: {step}")
                sys.exit()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    pred_x_all_steps_orig = mpmwrapper_learnable.simulator.x.to_numpy()
    pred_x_all_steps = pred_x_all_steps_orig[:, :(substep_pred * num_sim_steps + 1), :].transpose(1, 0, 2)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if cfg['train_cfg']['hou_vis']:
        visualize_simulation(trajectory=pred_x_all_steps)

if __name__=='__main__':
    main()