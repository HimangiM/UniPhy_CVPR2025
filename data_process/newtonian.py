import random
import os

current_path = os.getcwd()
local_path = '/'.join(current_path.strip().split('/')[:-1])

def falling_under_gravity(random_mu, random_lam):
    objects = ["sphere", "blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=newtonian visualization_cfg.num_frames=10 "
                  f"objects.{cur_obj}.geometry.pos_y=0.2 train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                  f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/newtonian_diverse/newtonian_{cur_obj}_mu{random_mu[idx]}_lam{random_lam[idx]}' "
                  f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]}")

    return

def horizontal_left(random_mu, random_lam):
    objects = ["sphere", "blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=newtonian visualization_cfg.num_frames=10 "
                  f"objects.{cur_obj}.geometry.pos_y=0.13 objects.{cur_obj}.geometry.pos_x=0 objects.{cur_obj}.material.velocity='[2., 0., 0.]' "
                  f"train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                  f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/newtonian_diverse/newtonian_{cur_obj}_horizontaleft_mu{random_mu[idx]}_lam{random_lam[idx]}' "
                  f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]}")
        return

def horizontal_right(random_mu, random_lam):
    objects = ["sphere", "blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    traj_direction = "horizontalright"
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=newtonian visualization_cfg.num_frames=10 "
                  f"objects.{cur_obj}.geometry.pos_y=0.13 objects.{cur_obj}.geometry.pos_x=0 objects.{cur_obj}.material.velocity='[-2., 0., 0.]' "
                  f"train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                  f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/newtonian_diverse/newtonian_{cur_obj}_{traj_direction}_mu{random_mu[idx]}_lam{random_lam[idx]}' "
                  f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]}")
        return

def diagonal_left(random_mu, random_lam):
    objects = ["sphere", "blobby", "torus", "blobby", "torus", "toy", "knurling", "pawn", "ellipsoid", "blobby", "knurling"] * 5
    traj_direction = "diagonalleft"
    for idx in range(len(random_mu)):
        cur_obj = objects[idx]
        os.system(f"python dataset_generate.py objects=[{cur_obj}] objects.{cur_obj}.geometry.num_particles=500 objects/{cur_obj}/material=newtonian visualization_cfg.num_frames=10 "
                      f"objects.{cur_obj}.geometry.pos_y=0.2 objects.{cur_obj}.geometry.pos_x=0 objects.{cur_obj}.material.velocity='[2., 0., 0.]' "
                      f"train_cfg.cuda_chunk_size=4096 train_cfg.particles_ti_root=1024 "
                      f"train_cfg.local_dir={local_path} train_cfg.save_dir='dataset/newtonian_diverse/newtonian_{cur_obj}_{traj_direction}_mu{random_mu[idx]}_lam{random_lam[idx]}' "
                      f"objects.{cur_obj}.material.mu={random_mu[idx]} objects.{cur_obj}.material.lam={random_lam[idx]}")

    return

if __name__ == '__main__':
    min_range_mu, max_range_mu = 50.0, 1000.0
    min_range_lam, max_range_lam = 30.0, 500000.0
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

    falling_under_gravity(random_mu=random_mu, random_lam=random_lam)
    horizontal_left(random_mu=random_mu, random_lam=random_lam)
    horizontal_right(random_mu=random_mu, random_lam=random_lam)
    diagonal_left(random_mu=random_mu, random_lam=random_lam)
