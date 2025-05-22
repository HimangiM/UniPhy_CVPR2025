import numpy as np
import torch
import os
from mpm_simulator_grad import MPMSimulatorConstitutive
from torch import nn
import taichi as ti
import argparse
import open3d as o3d
# from sdf_functions import generate_sdf_particles
import copy

@ti.data_oriented
class MPMWrapperLearnableStress:
    def __init__(self, objects, simulator_cfg, visualization_cfg, init_particles, target_particles, constitutive_function,
                 particles_ti_root, cuda_chunk_size, stress_model, tb_writer, stress_data_gt, embed_dim, fproj_model, scale_factor=None):

        self.objects = objects
        self.visualization_cfg = visualization_cfg
        n_dim = 3

        # Geometry
        self.init_particles = init_particles

        # Material
        # self.material = materials_obj
        self.material = None
        for mat_key in objects.keys():
            cur_mat_cfg = objects[mat_key]['material']
            cur_geo_cfg = objects[mat_key]['geometry']
            cur_material = None
            if cur_mat_cfg['material_type'] == "MPMSimulator.elasticity":
                cur_material = [MPMSimulatorConstitutive.elasticity for _ in range(self.init_particles.shape[0])]
            elif cur_mat_cfg['material_type'] == "MPMSimulator.von_mises":
                cur_material = [MPMSimulatorConstitutive.von_mises for _ in range(self.init_particles.shape[0])]
            elif cur_mat_cfg['material_type'] == "MPMSimulator.drucker_prager":
                cur_material = [MPMSimulatorConstitutive.drucker_prager for _ in range(self.init_particles.shape[0])]
            elif cur_mat_cfg['material_type'] == "MPMSimulator.viscous_fluid":
                cur_material = [MPMSimulatorConstitutive.viscous_fluid for _ in range(self.init_particles.shape[0])]
            self.material = cur_material if self.material is None else np.concatenate((self.material, cur_material), axis=0)
            self.material = np.array(self.material)

        self.init_velocities = None
        self.init_rhos = None
        self.init_mu = None
        self.init_lam = None
        self.init_alphas = None
        self.init_cohesion = None
        self.scale_factor = scale_factor

        if simulator_cfg['dtype'] == "float32":
            self.dtype = ti.f32
        self.dx = ti.field(self.dtype, shape=())
        self.inv_dx = ti.field(self.dtype, shape=())
        self.num_particles = ti.field(ti.i32, shape=())
        self.particle_rho = ti.field(dtype=self.dtype, needs_grad=True)
        # self.particle = ti.root.dynamic(ti.i, 2 ** 30, 1024)
        self.particle = ti.root.dynamic(ti.i, particles_ti_root, 256)
        self.particle.place(self.particle_rho, self.particle_rho.grad)
        self.cuda_chunk_size = cuda_chunk_size
        self.embed_dim = embed_dim

        for mat_key in objects.keys():
            cur_mat_cfg = objects[mat_key]['material']
            self.dt = 1. / cur_mat_cfg['inv_dt']  # TODO
            self.frame_dt = 1. / cur_mat_cfg['inv_frame_dt'] # TODO

        self.simulator = MPMSimulatorConstitutive(dtype=self.dtype, dt=self.dt, frame_dt=self.frame_dt, n_particles=self.num_particles,
                                      material=self.material, dx=self.dx, inv_dx=self.inv_dx,
                                      particle_layout=self.particle, gravity=simulator_cfg['gravity'],
                                      cuda_chunk_size=self.cuda_chunk_size, constitutive_function=constitutive_function,
                                      stress_model=stress_model, device="cuda", particles_ti_root=particles_ti_root,
                                      tb_writer=tb_writer, stress_gt=stress_data_gt, embed_dim=embed_dim,
                                      fproj_model=fproj_model)
        BC = simulator_cfg['BC']
        for bc in BC:
            if "ground" in bc:
                self.simulator.add_surface_collider(BC[bc][0], BC[bc][1], MPMSimulatorConstitutive.surface_sticky)
            elif "cylinder" in bc:
                self.simulator.add_cylinder_collider(BC[bc][0], BC[bc][1], BC[bc][2], MPMSimulatorConstitutive.surface_sticky)

        # Loss computation variables
        self.n_substeps = ti.field(ti.i32, shape=())
        self.n_substeps[None] = round(self.frame_dt / self.dt)
        self.target_particles = ti.Vector.field(n_dim, dtype=self.dtype, shape=(target_particles.shape[0], target_particles.shape[1]))  # P x T x 3
        self.target_particles.from_numpy(target_particles) # 500 x 59 x 3

    def initialize_particles(self):
        self.init_rhos = None
        self.init_velocities = None
        self.init_mu = None
        self.init_lam = None
        self.init_yield_stress = None
        self.init_plastic_viscosity = None
        self.init_friction_alpha = None
        self.init_cohesion = None
        n_particles = self.init_particles.shape[0]
        for mat_key in self.objects.keys():
            cur_mat_cfg = self.objects[mat_key]['material']
            cur_geo_cfg = self.objects[mat_key]['geometry']

            cur_rhos = np.repeat(cur_mat_cfg['rho'], n_particles).astype(np.float32)
            self.init_rhos = cur_rhos if self.init_rhos is None else np.concatenate((self.init_rhos, cur_rhos), axis=0)

            cur_velocities = np.tile(cur_mat_cfg['velocity'], (n_particles, 1)).astype(np.float32)
            # cur_velocities = np.tile([-0.1998, -0.5002, -0.1994], (n_particles, 1)).astype(np.float32)
            self.init_velocities = cur_velocities if self.init_velocities is None else np.concatenate((self.init_velocities, cur_velocities), axis=0)
            if self.scale_factor is not None:
                self.init_velocities = self.init_velocities * self.scale_factor
            else:
                print('Velocity with no scale factor', self.init_velocities[0])

            cur_mu = np.repeat(cur_mat_cfg['mu'], n_particles).astype(np.float32)
            self.init_mu = cur_mu if self.init_mu is None else np.concatenate((self.init_mu, cur_mu), axis=0)

            cur_lam = np.repeat(cur_mat_cfg['lam'], n_particles).astype(np.float32)
            self.init_lam = cur_lam if self.init_lam is None else np.concatenate((self.init_lam, cur_lam), axis=0)

            cur_yield_stress = np.repeat(cur_mat_cfg['yield_stress'], n_particles).astype(np.float32)
            self.init_yield_stress = cur_yield_stress if self.init_yield_stress is None else np.concatenate((self.init_yield_stress, cur_yield_stress), axis=0)

            cur_plastic_viscosity = np.repeat(cur_mat_cfg['plastic_viscosity'], n_particles).astype(np.float32)
            self.init_plastic_viscosity = cur_plastic_viscosity if self.init_plastic_viscosity is None else np.concatenate((self.init_plastic_viscosity, cur_plastic_viscosity), axis=0)

            cur_friction_alpha = np.repeat(cur_mat_cfg['friction_alpha'], n_particles).astype(np.float32)
            self.init_friction_alpha = cur_friction_alpha if self.init_friction_alpha is None else np.concatenate((self.init_friction_alpha, cur_friction_alpha), axis=0)

            cur_cohesion = np.repeat(0., n_particles).astype(np.float32)
            self.init_cohesion = cur_cohesion if self.init_cohesion is None else np.concatenate((self.init_cohesion, cur_cohesion), axis=0)

    def clear_grads(self):
        self.particle_rho.grad.fill(0)
        self.simulator.clear_grads()

    def clear_svd_grads(self):
        self.simulator.clear_svd_grads()

    @ti.kernel
    def update_stress(self, particle_stress: ti.types.ndarray(), s: ti.i32):
        ti.loop_config(parallelize=32, block_dim=16)
        for p in range(self.num_particles[None]):
            for i_stress in ti.static(range(3)):
                for j_stress in ti.static(range(3)):
                    self.simulator.stress[p, s][i_stress, j_stress] = particle_stress[p, i_stress, j_stress]

    @ti.kernel
    def update_traj_latent_learnable(self, traj_latent_values: ti.types.ndarray()):
        for e in range(self.embed_dim):
            self.simulator.traj_latent_ti[e] = traj_latent_values[e]

    @ti.kernel
    def set_loss_to_zero(self):
        self.simulator.loss[None] = 0.

    @ti.kernel
    def update_x_at_step(self, particles: ti.types.ndarray(), step: ti.i32):
        for p in range(self.num_particles[None]):
            for d in ti.static(range(3)):
                self.simulator.x[p, step][d] = particles[p, d]

    @ti.kernel
    def from_torch(self, particles: ti.types.ndarray(),
                   velocities: ti.types.ndarray(),
                   particle_rho: ti.types.ndarray(),
                   particle_mu: ti.types.ndarray(),
                   particle_lam: ti.types.ndarray(),
                   particle_yield_stress: ti.types.ndarray(),
                   particle_plastic_viscosity: ti.types.ndarray(),
                   particle_friction_alpha: ti.types.ndarray(),
                   particle_cohesion: ti.types.ndarray(),
                   material: ti.types.ndarray()
                   ):
        # assume cell is indexed by the bottom corner
        for p in range(self.num_particles[None]):
            self.particle_rho[p] = particle_rho[p]
            self.simulator.mu[p] = particle_mu[p]
            self.simulator.lam[p] = particle_lam[p]
            self.simulator.p_mass[p] = 0
            self.simulator.F[p, 0] = ti.Matrix.identity(self.dtype, 3)
            self.simulator.F_tmp[p, 0] = ti.Matrix.identity(self.dtype, 3)
            self.simulator.C[p, 0] = ti.Matrix.zero(self.dtype, 3, 3)
            self.simulator.stress[p, 0] = ti.Matrix.zero(self.dtype, 3, 3)
            for d in ti.static(range(3)):
                self.simulator.x[p, 0][d] = particles[p, d]
                self.simulator.v[p, 0][d] = velocities[p, d]
            self.simulator.yield_stress[p] = particle_yield_stress[p]
            self.simulator.plastic_viscosity[p] = particle_plastic_viscosity[p]
            self.simulator.friction_alpha[p] = particle_friction_alpha[p]
            self.simulator.cohesion[p] = particle_cohesion[p]
            self.simulator.material[p] = material[p]

    @ti.kernel
    def compute_particle_mass(self):
        for p in range(self.num_particles[None]):
            self.simulator.p_mass[p] = self.particle_rho[p] * self.simulator.p_vol[None]

    def simulator_variables_initialize(self):
        torch.cuda.synchronize()
        ti.sync()
        self.num_particles[None] = self.init_particles.shape[0]
        self.dx[None], self.inv_dx[None] = 0.02, 50
        self.simulator.p_vol[None] = (self.dx[None] * 0.5) ** 3
        self.clear_grads()
        self.simulator.cached_states.clear()
        self.from_torch(
                        self.init_particles,
                        self.init_velocities,
                        self.init_rhos,
                        self.init_mu,
                        self.init_lam,
                        self.init_yield_stress,
                        self.init_plastic_viscosity,
                        self.init_friction_alpha,
                        self.init_cohesion,
                        self.material
                        )
        self.compute_particle_mass()
        self.set_loss_to_zero()
        self.simulator.cfl_satisfy[None] = True

    @ti.kernel
    def compute_loss(self):
        for f_step in range((self.visualization_cfg['num_frames'] * self.simulator.n_substeps[None])):
            local_index = (f_step+1) % self.cuda_chunk_size
            for p in range(self.num_particles[None]):
                self.simulator.loss[None] += (((((self.simulator.x[p, local_index]) - self.target_particles[p, local_index]) ** 2).sum() / self.num_particles[None]))