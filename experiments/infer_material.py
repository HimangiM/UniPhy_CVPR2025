import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from tqdm.autonotebook import tqdm, trange
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import warp as wp

import nclaw
from nclaw.data import MPMDataset
from nclaw.sim import MPMModelBuilder, MPMCacheDiffSim, MPMStaticsInitializer, MPMInitData
from nclaw.utils import get_root
from sklearn.cluster import KMeans

root: Path = get_root(__file__)
print ('root:', root)

@hydra.main(version_base='1.2', config_path=str(root / 'configs'), config_name='train')
def main(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(True)
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

    # path
    # ckpt_stage1_dir = f'/data18/hmittal/azs_221_166/trained_models/latent_nclaw/train_allmaterials_stressFC5L5L_hs128_embed32_100traj_1e3_jellyfinetune2'
    ckpt_stage1_dir = cfg.env.blob.material.ckpt
    print ("Ckpt loaded from:", ckpt_stage1_dir)

    log_root: Path = root / 'log'
    exp_root: Path = log_root / cfg.name / f'ckpt_{cfg.env.blob.material.name}'
    nclaw.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    OmegaConf.save(cfg, exp_root / 'hydra.yaml', resolve=True)
    writer = SummaryWriter(exp_root, purge_step=0)

    ckpt_root: Path = exp_root / 'ckpt'
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # traj_dir = 'jelly_3/jelly_100000_0.2'
    # dataset_root: Path = Path(f'/data18/hmittal/azs_221_166/dataset/{traj_dir}/dataset')
    dataset_root: Path = Path(cfg.env.blob.material.traj_dir)

    # data
    dataset = MPMDataset(dataset_root, torch_device)

    # warp

    model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp_device, requires_grad=True)
    sim = MPMCacheDiffSim(model, cfg.sim.num_steps)
    statics_initializer = MPMStaticsInitializer(model)
    init_data = MPMInitData.get(cfg.env.blob)
    statics_initializer.add_group(init_data)
    statics = statics_initializer.finalize()

    # material

    elasticity_requires_grad = cfg.env.blob.material.elasticity.requires_grad
    plasticity_requires_grad = cfg.env.blob.material.plasticity.requires_grad

    elasticity: nn.Module = getattr(nclaw.material, cfg.env.blob.material.elasticity.cls)(hidden_size=128, embed_dim=32, normalize_input=True)
    elasticity.to(torch_device)
    if len(list(elasticity.parameters())) == 0:
        elasticity_requires_grad = False
    elasticity.requires_grad_(elasticity_requires_grad)
    elasticity.train(elasticity_requires_grad)

    plasticity: nn.Module = getattr(nclaw.material, cfg.env.blob.material.plasticity.cls)(hidden_size=128, embed_dim=32, normalize_input=True, alpha=0.001)
    plasticity.to(torch_device)
    if len(list(plasticity.parameters())) == 0:
        plasticity_requires_grad = False
    plasticity.requires_grad_(plasticity_requires_grad)
    plasticity.train(plasticity_requires_grad)

    ckpt = torch.load(f'{ckpt_stage1_dir}/model_30.pth', map_location=torch_device)
    elasticity.load_state_dict(ckpt['stress_model_state_dict'])
    plasticity.load_state_dict(ckpt['plasticity_model_state_dict'])

    print ("Kmeans-centroid latent loading....")
    traj_path = f'{ckpt_stage1_dir}/traj_latent_30.pth'
    trajectory_latent_embedding_orig = torch.load(traj_path).weight.detach()
    print ('Shape: ', trajectory_latent_embedding_orig.shape)
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(trajectory_latent_embedding_orig.cpu().numpy())
    centroids = kmeans.cluster_centers_[2]
    trajectory_latent_embedding = torch.Tensor(centroids).unsqueeze(0)
    trajectory_latent = torch.nn.Embedding.from_pretrained(trajectory_latent_embedding, freeze=False).to(torch_device)

    torch.save({
        'elasticity': elasticity.state_dict(),
        'plasticity': plasticity.state_dict(),
        'trajectory_latent': trajectory_latent,
    }, ckpt_root / f'{0:04d}.pt')

    if elasticity_requires_grad:
        elasticity_optimizer = torch.optim.Adam(elasticity.parameters(), lr=cfg.train.elasticity_lr, weight_decay=cfg.train.elasticity_wd)
        elasticity_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=elasticity_optimizer, T_max=cfg.train.num_epochs)
    if plasticity_requires_grad:
        plasticity_optimizer = torch.optim.Adam(plasticity.parameters(), lr=cfg.train.plasticity_lr, weight_decay=cfg.train.plasticity_wd)
        plasticity_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=plasticity_optimizer, T_max=cfg.train.num_epochs)

    # latent_optimizer = torch.optim.Adam(trajectory_latent.parameters(), lr=0.1, weight_decay=cfg.train.elasticity_wd)
    latent_optimizer = torch.optim.AdamW(trajectory_latent.parameters(), lr=cfg.train.latent_lr, weight_decay=cfg.train.elasticity_wd)
    latent_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=latent_optimizer, T_max=cfg.train.num_epochs)

    criterion = nn.MSELoss()
    criterion.to(torch_device)

    for epoch in trange(cfg.train.num_epochs, position=1):

        if elasticity_requires_grad:
            elasticity_optimizer.zero_grad()
        if plasticity_requires_grad:
            plasticity_optimizer.zero_grad()

        latent_optimizer.zero_grad()

        losses = defaultdict(int)

        x, v, C, F, _ = dataset[0]
        traj_ids = 0
        for step in tqdm(range(cfg.sim.num_steps), position=0, leave=False):
            stress = elasticity(F, C, trajectory_latent, traj_ids)
            x, v, C, F = sim(statics, step, x, v, C, F, stress)
            F = plasticity(F, trajectory_latent, traj_ids)
            xt, vt, Ct, Ft, _ = dataset[step + 1]
            loss_acc = criterion(x, xt)
            losses['acc'] += loss_acc
        loss = sum(losses.values())

        loss.backward()

        if elasticity_requires_grad:
            elasticity_grad_norm = clip_grad_norm_(
                elasticity.parameters(),
                max_norm=cfg.train.elasticity_grad_max_norm,
                error_if_nonfinite=True)
            elasticity_optimizer.step()

        if plasticity_requires_grad:
            plasticity_grad_norm = clip_grad_norm_(
                plasticity.parameters(),
                max_norm=cfg.train.plasticity_grad_max_norm,
                error_if_nonfinite=True)
            plasticity_optimizer.step()

        latent_grad = clip_grad_norm_(
                trajectory_latent.parameters(),
                max_norm=cfg.train.elasticity_grad_max_norm,
                error_if_nonfinite=True)
        latent_optimizer.step()

        msgs = [
            cfg.name,
            time.strftime('%H:%M:%S'),
            'epoch {:{width}d}/{}'.format(epoch + 1, cfg.train.num_epochs, width=len(str(cfg.train.num_epochs)))
        ]

        if elasticity_requires_grad:
            elasticity_lr = elasticity_optimizer.param_groups[0]['lr']
            msgs.extend([
                'e-lr {:.2e}'.format(elasticity_lr),
                'e-|grad| {:.4f}'.format(elasticity_grad_norm),
            ])

        if plasticity_requires_grad:
            plasticity_lr = plasticity_optimizer.param_groups[0]['lr']
            msgs.extend([
                'p-lr {:.2e}'.format(plasticity_lr),
                'p-|grad| {:.4f}'.format(plasticity_grad_norm),
            ])

        latent_lr = latent_optimizer.param_groups[0]['lr']
        msgs.extend([
            'l-lr {:.2e}'.format(latent_lr),
            'l-|grad| {:.4f}'.format(latent_grad),
        ])

        for loss_k, loss_v in losses.items():
            msgs.append('{} {:.4f}'.format(loss_k, loss_v.item()))
            writer.add_scalar('loss/{}'.format(loss_k), loss_v.item(), epoch + 1)

        msg = ','.join(msgs)
        tqdm.write('[{}]'.format(msg))

        torch.save({
            'elasticity': elasticity.state_dict(),
            'plasticity': plasticity.state_dict(),
            'trajectory_latent': trajectory_latent,
        }, ckpt_root / '{:04d}.pt'.format(epoch + 1))

        if elasticity_requires_grad:
            elasticity_lr = elasticity_optimizer.param_groups[0]['lr']
            elasticity_lr_scheduler.step()
        if plasticity_requires_grad:
            plasticity_lr = plasticity_optimizer.param_groups[0]['lr']
            plasticity_lr_scheduler.step()

        latent_lr_scheduler.step()

    writer.close()


if __name__ == '__main__':
    main()
