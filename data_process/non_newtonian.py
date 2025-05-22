import random
import os

current_path = os.getcwd()
local_path = '/'.join(current_path.strip().split('/')[:-1])

def falling_under_gravity(random_mu, random_lam, random_yield_stress, random_plastic_viscosity):
    objects = ["blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        save_file_name = f"non_newtonian_{cur_obj}_mu{random_mu[idx]}_lam{random_lam[idx]}_ys{random_yield_stress[idx]}_pv{random_plastic_viscosity[idx]}"
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=non_newtonian visualization_cfg.num_frames=10 "
                  f"objects.{cur_obj}.geometry.pos_y=0.2 objects.{cur_obj}.geometry.pos_x=0 objects.{cur_obj}.material.velocity='[-0.1998, -3.500, -0.1994]' "
                  f"train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                  f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/non_newtonian_diverse/{save_file_name}' "
                  f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]} "
                  f"objects.{cur_obj}.material.yield_stress={random_yield_stress[idx]} objects.{cur_obj}.material.plastic_viscosity={random_plastic_viscosity[idx]}")
    return

def horizontal_left(random_mu, random_lam, random_yield_stress, random_plastic_viscosity):
    objects = ["blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        save_file_name = f"non_newtonian_{cur_obj}_horizontaleft_mu{random_mu[idx]}_lam{random_lam[idx]}_ys{random_yield_stress[idx]}_pv{random_plastic_viscosity[idx]}"
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=non_newtonian visualization_cfg.num_frames=10 "
                  f"objects.{cur_obj}.geometry.pos_y=0.13 objects.{cur_obj}.geometry.pos_x=0 objects.{cur_obj}.material.velocity='[2., 0., 0.]' "
                  f"train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                  f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/non_newtonian_diverse/{save_file_name}' "
                  f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]} "
                  f"objects.{cur_obj}.material.yield_stress={random_yield_stress[idx]} objects.{cur_obj}.material.plastic_viscosity={random_plastic_viscosity[idx]}")
    return

def horizontal_right(random_mu, random_lam, random_yield_stress, random_plastic_viscosity):
    objects = ["blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        save_file_name = f"non_newtonian_{cur_obj}_horizontalright_mu{random_mu[idx]}_lam{random_lam[idx]}_ys{random_yield_stress[idx]}_pv{random_plastic_viscosity[idx]}"
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=non_newtonian visualization_cfg.num_frames=10 "
                  f"objects.{cur_obj}.geometry.pos_y=0.13 objects.{cur_obj}.geometry.pos_x=0 objects.{cur_obj}.material.velocity='[-2., 0., 0.]' "
                  f"train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                  f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/non_newtonian_diverse/{save_file_name}' "
                  f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]} "
                  f"objects.{cur_obj}.material.yield_stress={random_yield_stress[idx]} objects.{cur_obj}.material.plastic_viscosity={random_plastic_viscosity[idx]}")
    return

def diagonal_left(random_mu, random_lam, random_yield_stress, random_plastic_viscosity):
    objects = ["blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        save_file_name = f"non_newtonian_{cur_obj}_diagonalleft_mu{random_mu[idx]}_lam{random_lam[idx]}_ys{random_yield_stress[idx]}_pv{random_plastic_viscosity[idx]}"
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=non_newtonian visualization_cfg.num_frames=10 "
                  f"objects.{cur_obj}.geometry.pos_y=0.2 objects.{cur_obj}.geometry.pos_x=0 objects.{cur_obj}.material.velocity='[2., 0., 0.]' "
                  f"train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                  f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/non_newtonian_diverse/{save_file_name}' "
                  f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]} "
                  f"objects.{cur_obj}.material.yield_stress={random_yield_stress[idx]} objects.{cur_obj}.material.plastic_viscosity={random_plastic_viscosity[idx]}")
    return

if __name__ == '__main__':
    min_range_mu, max_range_mu = 1e3, 2e6
    min_range_lam, max_range_lam = 1e3, 2e6
    min_range_ys, max_range_ys = 1e3, 2e6
    min_range_pv, max_range_pv = 0.1, 100.0
    num_samples = 1

    random_mu = []
    while len(random_mu) < num_samples:
        r_mu = random.uniform(min_range_mu, max_range_mu)
        if r_mu not in random_mu:
            random_mu.append(r_mu)
            print(r_mu)

    random_lam = []
    while len(random_lam) < num_samples:
        r_lam = random.uniform(min_range_lam, max_range_lam)
        if r_lam not in random_lam:
            random_lam.append(r_lam)
            print(r_lam)

    random_ys = []
    while len(random_ys) < num_samples:
        r_ys = random.uniform(min_range_ys, max_range_ys)
        if r_ys not in random_ys:
            random_ys.append(r_ys)
            print(r_ys)

    random_pv = []
    while len(random_pv) < num_samples:
        r_pv = random.uniform(min_range_pv, max_range_pv)
        if r_pv not in random_pv:
            random_pv.append(r_pv)
            print(r_pv)

    falling_under_gravity(random_mu=random_mu, random_lam=random_lam, random_yield_stress=random_ys, random_plastic_viscosity=random_pv)
    horizontal_left(random_mu=random_mu, random_lam=random_lam, random_yield_stress=random_ys, random_plastic_viscosity=random_pv)
    horizontal_right(random_mu=random_mu, random_lam=random_lam, random_yield_stress=random_ys, random_plastic_viscosity=random_pv)
    diagonal_left(random_mu=random_mu, random_lam=random_lam, random_yield_stress=random_ys, random_plastic_viscosity=random_pv)