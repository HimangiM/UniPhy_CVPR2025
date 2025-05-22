import random
import os

current_path = os.getcwd()
local_path = '/'.join(current_path.strip().split('/')[:-1])

def falling_under_gravity(random_mu, random_lam, random_yield_stress):
    objects = ["sphere", "blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=plasticine visualization_cfg.num_frames=10 "
                  f"objects.{cur_obj}.geometry.pos_y=0.2 train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                  f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/plasticine_diverse/plasticine_{cur_obj}_mu{random_mu[idx]}_lam{random_lam[idx]}_ys{random_yield_stress[idx]}' "
                  f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]} objects.{cur_obj}.material.yield_stress={random_yield_stress[idx]}")
    return

def horizontal_left(random_mu, random_lam, random_yield_stress):
    objects = ["sphere", "blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    traj_dirn = "horizontalleft"
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        save_file_name = f"plasticine_{cur_obj}_{traj_dirn}_mu{random_mu[idx]}_lam{random_lam[idx]}_ys{random_yield_stress[idx]}"
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=plasticine visualization_cfg.num_frames=10 "
                  f"objects.{cur_obj}.geometry.pos_y=0.11 objects.{cur_obj}.geometry.pos_x=0 objects.{cur_obj}.material.velocity='[2., 0., 0.]' "
                  f"train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                  f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/plasticine_diverse/{save_file_name}' "
                  f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]} objects.{cur_obj}.material.yield_stress={random_yield_stress[idx]}")
    return

def horizontal_right(random_mu, random_lam, random_yield_stress):
    objects = ["sphere", "blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    traj_dirn = "horizontalright"
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        save_file_name = f"plasticine_{cur_obj}_{traj_dirn}_mu{random_mu[idx]}_lam{random_lam[idx]}_ys{random_yield_stress[idx]}"
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=plasticine visualization_cfg.num_frames=10 "
                  f"objects.{cur_obj}.geometry.pos_y=0.11 objects.{cur_obj}.geometry.pos_x=0 objects.{cur_obj}.material.velocity='[-2., 0., 0.]' "
                  f"train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                  f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/plasticine_diverse/{save_file_name}' "
                  f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]} objects.{cur_obj}.material.yield_stress={random_yield_stress[idx]}")
    return

def diagonal_left(random_mu, random_lam, random_yield_stress):
    objects = ["sphere", "blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    traj_dirn = "diagonalleft"
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        save_file_name = f"plasticine_{cur_obj}_{traj_dirn}_mu{random_mu[idx]}_lam{random_lam[idx]}_ys{random_yield_stress[idx]}"
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=plasticine visualization_cfg.num_frames=10 "
                      f"objects.{cur_obj}.geometry.pos_y=0.2 objects.{cur_obj}.geometry.pos_x=0 objects.{cur_obj}.material.velocity='[2., 0., 0.]' "
                      f"train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                      f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/plasticine_diverse/{save_file_name}' "
                      f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]} objects.{cur_obj}.material.yield_stress={random_yield_stress[idx]}")
    return

if __name__ == '__main__':
    min_range_mu, max_range_mu = 10000.0, 1000000.0
    min_range_lam, max_range_lam = 10000.0, 3000000.0
    min_range_yield_stress, max_range_yield_stress = 5000.0, 10000.0
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

    random_yield_stress = []
    while len(random_yield_stress) < num_samples:
        r_yield_stress = random.uniform(min_range_yield_stress, max_range_yield_stress)
        if r_yield_stress not in random_yield_stress:
            random_yield_stress.append(r_yield_stress)
            print(r_yield_stress)

    falling_under_gravity(random_mu=random_mu, random_lam=random_lam, random_yield_stress=random_yield_stress)
    horizontal_left(random_mu=random_mu, random_lam=random_lam, random_yield_stress=random_yield_stress)
    horizontal_right(random_mu=random_mu, random_lam=random_lam, random_yield_stress=random_yield_stress)
    diagonal_left(random_mu=random_mu, random_lam=random_lam, random_yield_stress=random_yield_stress)