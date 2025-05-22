from pathlib import Path
import random
from tqdm import tqdm, trange
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn
import warp as wp

import nclaw
from nclaw.material import ComposeMaterial, ComposeMaterialLatent
from nclaw.sim import MPMModelBuilder, MPMStateInitializer, MPMStaticsInitializer, MPMInitData, MPMForwardSim
from nclaw.utils import get_root, sample_vel

root: Path = get_root(__file__)

@torch.no_grad()
@hydra.main(version_base='1.2', config_path=str(root / 'configs'), config_name='eval')
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg, resolve=True))

    # init

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    wp.init()
    wp_device = wp.get_device(f'cuda:{cfg.gpu}')
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})

    torch_device = torch.device(f'cuda:{cfg.gpu}')
    torch.backends.cudnn.benchmark = True

    requires_grad = False

    # path

    log_root: Path = root / 'log'
    exp_root: Path = log_root / cfg.name
    state_root: Path = exp_root / 'state'
    nclaw.utils.mkdir(state_root, overwrite=cfg.overwrite, resume=cfg.resume)
    OmegaConf.save(cfg, exp_root / 'hydra.yaml', resolve=True)

    # warp

    model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp_device, requires_grad)
    sim = MPMForwardSim(model)
    state_initializer = MPMStateInitializer(model)
    statics_initializer = MPMStaticsInitializer(model)

    # env

    elasticities = []
    plasticities = []

    for blob, blob_cfg in sorted(cfg.env.items()):

        if blob in ['name', 'start', 'end']:
            continue

        # material

        elasticity: nn.Module = getattr(nclaw.material, blob_cfg.material.elasticity.cls)(hidden_size=128, embed_dim=32, normalize_input=True)
        elasticity.to(torch_device)
        plasticity: nn.Module = getattr(nclaw.material, blob_cfg.material.plasticity.cls)(hidden_size=128, embed_dim=32, normalize_input=True, alpha=0.001)
        plasticity.to(torch_device)

        if ckpt_path := blob_cfg.material.ckpt_latent:
            print (f"Ckpt path loaded: {ckpt_path}")
            ckpt = torch.load(log_root / ckpt_path, map_location=torch_device)
            elasticity.load_state_dict(ckpt['elasticity'])
            plasticity.load_state_dict(ckpt['plasticity'])
            trajectory_latent = torch.nn.Embedding.from_pretrained(ckpt['trajectory_latent'].weight, freeze=False).to(torch_device)

        init_data = MPMInitData.get(blob_cfg)

        if blob_cfg.vel.random:
            lin_vel, ang_vel = sample_vel(blob_cfg.vel)
        else:
            lin_vel = np.array(blob_cfg.vel.lin_vel)
            ang_vel = np.array(blob_cfg.vel.ang_vel)
        init_data.set_lin_vel(lin_vel)
        init_data.set_ang_vel(ang_vel)

        state_initializer.add_group(init_data)
        statics_initializer.add_group(init_data)
        elasticities.append(elasticity)
        plasticities.append(plasticity)

    state, sections = state_initializer.finalize()
    statics = statics_initializer.finalize()
    x, v, C, F, stress = state.to_torch()

    types = torch.zeros(sum(sections), dtype=torch.int)
    for i, (left, right) in enumerate(zip(np.cumsum(sections)[:-1], np.cumsum(sections)[1:])):
        types[left:right] = i
    ckpt = dict(x=x, v=v, C=C, F=F, stress=stress, sections=sections, types=types)
    torch.save(ckpt, state_root / f'{0:04d}.pt')

    # elasticity = ComposeMaterialLatent(elasticities, sections)  # TODO
    elasticity.to(torch_device)
    elasticity.requires_grad_(requires_grad)
    elasticity.eval()

    # plasticity = ComposeMaterialLatent(plasticities, sections) # TODO
    plasticity.to(torch_device)
    plasticity.requires_grad_(requires_grad)
    plasticity.eval()

    if ckpt_path := blob_cfg.material.ckpt:
        trajectory_latent.eval()

    traj_ids = 0
    for step in trange(1, cfg.sim.num_steps + 1):
        if cfg.dataset:
            Fe = F.clone()
        stress = elasticity(F, C, trajectory_latent, traj_ids)
        state.from_torch(stress=stress)
        x, v, C, F = sim(statics, state)
        if cfg.dataset:
            Fp = F.clone()
        F = plasticity(F, trajectory_latent, traj_ids)
        state.from_torch(F=F)
        statics_initializer.update(statics, step)
        ckpt = dict(x=x, v=v, C=C, F=F, stress=stress, sections=sections, types=types)
        if cfg.dataset:
            ckpt['Fe'] = Fe
            ckpt['Fp'] = Fp
        if step % cfg.sim.skip_frame == 0:
            torch.save(ckpt, state_root / f'{step:04d}.pt')


if __name__ == '__main__':
    main()