from mpmwrapper_perparticleparams import MPMWrapper
import numpy as np
import taichi as ti
import hydra
import omegaconf
import torch
import time
import os
from omegaconf import OmegaConf

torch.autograd.set_detect_anomaly(True)

@hydra.main(config_path='configs', config_name='default', version_base=None)
def main(cfg: omegaconf.DictConfig):
    start_time = time.time()
    ti_mem_fraction = 0.5
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=ti_mem_fraction, debug=True)
    save_dir = cfg['train_cfg']['save_dir']
    device = "cuda"

    num_sim_steps = cfg['visualization_cfg']['num_frames']
    cuda_chunk_size = cfg['train_cfg']['cuda_chunk_size']
    particles_ti_root = cfg['train_cfg']['particles_ti_root']
    local_dir = cfg['train_cfg']['local_dir']

    mpmwrapper_gt = MPMWrapper(cfg['objects'], cfg['simulator_cfg'], particles_ti_root=particles_ti_root, cuda_chunk_size=cuda_chunk_size)
    mpmwrapper_gt.initialize_particles()
    mpmwrapper_gt.simulator_variables_initialize()

    substep_gt = mpmwrapper_gt.simulator.n_substeps[None]

    particle_x = torch.zeros([1024, 4096, 3], dtype=torch.float32).to(device)
    particle_v = torch.zeros([1024, 4096, 3], dtype=torch.float32).to(device)
    particle_C = torch.zeros([1024, 4096, 3, 3], dtype=torch.float32).to(device)
    particle_F = torch.zeros([1024, 4096, 3, 3], dtype=torch.float32).to(device)
    particle_Ftmp = torch.zeros([1024, 4096, 3, 3], dtype=torch.float32).to(device)
    particle_stress = torch.zeros([1024, 4096, 3, 3], dtype=torch.float32).to(device)

    # Simulation
    cfl_flag = False
    for f in range(num_sim_steps):
        print ("Step: ", f)

        for i in range(substep_gt * f, substep_gt * (f + 1)):
            if mpmwrapper_gt.simulator.cfl_satisfy[None]:
                mpmwrapper_gt.simulator.substep(i)

            if not mpmwrapper_gt.simulator.cfl_satisfy[None]:
                mpmwrapper_gt.simulator.cached_states.clear()
                print ("Cfl not satisfied")
                cfl_flag = True

            mpmwrapper_gt.simulator.get_x_gt(i, particle_x)
            mpmwrapper_gt.simulator.get_v_gt(i, particle_v)
            mpmwrapper_gt.simulator.get_C(i, particle_C)
            mpmwrapper_gt.simulator.get_F(i+1, particle_F)
            mpmwrapper_gt.simulator.get_Ftmp(i+1, particle_Ftmp)
            mpmwrapper_gt.simulator.get_stress(i+1, particle_stress)

    # Save states
    if cfl_flag == False:
        save_path = os.path.join(local_dir, save_dir)

        for k in cfg['objects']:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "config.yaml"), 'w') as cfg_file:
                OmegaConf.save(cfg, cfg_file)

            final_x = particle_x[:mpmwrapper_gt.num_particles[None], :(num_sim_steps * substep_gt + 2)]
            final_v = particle_v[:mpmwrapper_gt.num_particles[None], :(num_sim_steps * substep_gt + 2)]
            final_C = particle_C[:mpmwrapper_gt.num_particles[None], :(num_sim_steps * substep_gt + 2)]
            final_F = particle_F[:mpmwrapper_gt.num_particles[None], :(num_sim_steps * substep_gt + 2)][:, :]
            final_Ftmp = particle_Ftmp[:mpmwrapper_gt.num_particles[None], :(num_sim_steps * substep_gt + 2)]
            final_stress = particle_stress[:mpmwrapper_gt.num_particles[None], :(num_sim_steps * substep_gt + 2)][:, :]

            x_gt = final_x.permute(1, 0, 2)
            v_gt = final_v.permute(1, 0, 2)
            C_gt = final_C.permute(1, 0, 2, 3)
            F_gt = final_F.permute(1, 0, 2, 3)
            Ftmp_gt = final_Ftmp.permute(1, 0, 2, 3)
            stress_gt = final_stress.permute(1, 0, 2, 3)

            torch.save(x_gt, f"{save_path}/GtX.pt")
            torch.save(v_gt, f"{save_path}/GtV.pt")
            torch.save(C_gt, f"{save_path}/GtC.pt")
            torch.save(F_gt, f"{save_path}/GtF.pt")
            torch.save(Ftmp_gt, f"{save_path}/GtFtmp.pt")
            torch.save(stress_gt, f"{save_path}/GtStress.pt")

        print (f"Simulated saved at {save_path}")
    else:
        print ("CFL violated")

if __name__=='__main__':
    main()








