import numpy as np
import open3d
import torch
from mpm_simulator_perparticleparams import MPMSimulator
import taichi as ti
import argparse
import open3d as o3d
from sdf_functions import generate_sdf_particles
import copy

@ti.data_oriented
class MPMWrapper:
    def __init__(self, objects, simulator_cfg, particles_ti_root, cuda_chunk_size):

        self.objects = objects

        # Geometry
        self.init_particles = None
        for geo_key in objects.keys():
            cur_geo_cfg = objects[geo_key]['geometry']
            cur_object_particles = generate_sdf_particles(cur_geo_cfg['sdf_func'], cur_geo_cfg['sdf_params'])
            res = np.random.choice(cur_object_particles.shape[0], cur_geo_cfg['num_particles'])
            cur_object_particles_sampled = cur_object_particles[res, :]
            cur_object_particles_transformed = self.rotate_translate_points(cur_object_particles_sampled,
                                                                            pos_x=cur_geo_cfg['pos_x'],
                                                                            pos_y=cur_geo_cfg['pos_y'],
                                                                            pos_z=cur_geo_cfg['pos_z'],
                                                                            rot_x=cur_geo_cfg['rot_x'],
                                                                            rot_y=cur_geo_cfg['rot_y'],
                                                                            rot_z=cur_geo_cfg['rot_z'])
            self.init_particles = cur_object_particles_transformed if self.init_particles is None else np.concatenate((self.init_particles, cur_object_particles_transformed), axis=0, dtype=np.float32)

        # Material
        self.material = None
        for mat_key in objects.keys():
            cur_mat_cfg = objects[mat_key]['material']
            cur_geo_cfg = objects[mat_key]['geometry']
            cur_material = None
            if cur_mat_cfg['material_type'] == "MPMSimulator.elasticity":
                cur_material = [MPMSimulator.elasticity for _ in range(cur_geo_cfg['num_particles'])]
            elif cur_mat_cfg['material_type'] == "MPMSimulator.von_mises":
                cur_material = [MPMSimulator.von_mises for _ in range(cur_geo_cfg['num_particles'])]
            elif cur_mat_cfg['material_type'] == "MPMSimulator.drucker_prager":
                cur_material = [MPMSimulator.drucker_prager for _ in range(cur_geo_cfg['num_particles'])]
            elif cur_mat_cfg['material_type'] == "MPMSimulator.viscous_fluid":
                cur_material = [MPMSimulator.viscous_fluid for _ in range(cur_geo_cfg['num_particles'])]
            self.material = cur_material if self.material is None else np.concatenate((self.material, cur_material), axis=0)
            self.material = np.array(self.material)

        self.init_velocities = None
        self.init_rhos = None
        self.init_mu = None
        self.init_lam = None
        self.init_alphas = None
        self.init_cohesion = None

        if simulator_cfg['dtype'] == "float32":
            self.dtype = ti.f32
        self.dx = ti.field(self.dtype, shape=())
        self.inv_dx = ti.field(self.dtype, shape=())
        self.num_particles = ti.field(ti.i32, shape=())
        self.particle_rho = ti.field(dtype=self.dtype)
        # self.particle = ti.root.dynamic(ti.i, 2 ** 30, 1024)
        self.particle = ti.root.dynamic(ti.i, particles_ti_root, cuda_chunk_size)
        self.particle.place(self.particle_rho)
        self.cuda_chunk_size = cuda_chunk_size

        for mat_key in objects.keys():
            cur_mat_cfg = objects[mat_key]['material']
            self.dt = 1. / cur_mat_cfg['inv_dt']
            self.frame_dt = 1. / cur_mat_cfg['inv_frame_dt']

        self.simulator = MPMSimulator(dtype=self.dtype, dt=self.dt, frame_dt=self.frame_dt, n_particles=self.num_particles,
                                      material=self.material, dx=self.dx, inv_dx=self.inv_dx,
                                      particle_layout=self.particle, gravity=simulator_cfg['gravity'],
                                      cuda_chunk_size=self.cuda_chunk_size)
        BC = simulator_cfg['BC']
        for bc in BC:
            if "ground" in bc:
                self.simulator.add_surface_collider(BC[bc][0], BC[bc][1], MPMSimulator.surface_sticky)
            elif "cylinder" in bc:
                self.simulator.add_cylinder_collider(BC[bc][0], BC[bc][1], BC[bc][2], MPMSimulator.surface_sticky)

    def load_points(self, path, num_points):
        all_points = []
        sample_idx = None
        for i in range(14):
            cur_particles = torch.Tensor(open3d.io.read_point_cloud(f'{path}/{i}.ply').points).to('cuda')
            if sample_idx is None:
                sample_idx = np.random.choice(cur_particles.shape[0], num_points)
            cur_particles = cur_particles[sample_idx]
            all_points.append(cur_particles)

        all_points = torch.stack(all_points).to('cuda')
        print("Loaded internal filled points from: ", path)
        return all_points

    def initialize_particles(self):
        self.init_rhos = None
        self.init_velocities = None
        self.init_mu = None
        self.init_lam = None
        self.init_yield_stress = None
        self.init_plastic_viscosity = None
        self.init_friction_alpha = None
        self.init_cohesion = None
        for mat_key in self.objects.keys():
            cur_mat_cfg = self.objects[mat_key]['material']
            cur_geo_cfg = self.objects[mat_key]['geometry']

            cur_rhos = np.repeat(cur_mat_cfg['rho'], cur_geo_cfg['num_particles']).astype(np.float32)
            self.init_rhos = cur_rhos if self.init_rhos is None else np.concatenate((self.init_rhos, cur_rhos), axis=0)

            cur_velocities = np.tile(cur_mat_cfg['velocity'], (cur_geo_cfg['num_particles'], 1)).astype(np.float32)
            self.init_velocities = cur_velocities if self.init_velocities is None else np.concatenate((self.init_velocities, cur_velocities), axis=0)

            cur_mu = np.repeat(cur_mat_cfg['mu'], cur_geo_cfg['num_particles']).astype(np.float32)
            self.init_mu = cur_mu if self.init_mu is None else np.concatenate((self.init_mu, cur_mu), axis=0)

            cur_lam = np.repeat(cur_mat_cfg['lam'], cur_geo_cfg['num_particles']).astype(np.float32)
            self.init_lam = cur_lam if self.init_lam is None else np.concatenate((self.init_lam, cur_lam), axis=0)

            cur_yield_stress = np.repeat(cur_mat_cfg['yield_stress'], cur_geo_cfg['num_particles']).astype(np.float32)
            self.init_yield_stress = cur_yield_stress if self.init_yield_stress is None else np.concatenate((self.init_yield_stress, cur_yield_stress), axis=0)

            cur_plastic_viscosity = np.repeat(cur_mat_cfg['plastic_viscosity'], cur_geo_cfg['num_particles']).astype(np.float32)
            self.init_plastic_viscosity = cur_plastic_viscosity if self.init_plastic_viscosity is None else np.concatenate((self.init_plastic_viscosity, cur_plastic_viscosity), axis=0)

            cur_friction_alpha = np.repeat(cur_mat_cfg['friction_alpha'], cur_geo_cfg['num_particles']).astype(np.float32)
            self.init_friction_alpha = cur_friction_alpha if self.init_friction_alpha is None else np.concatenate((self.init_friction_alpha, cur_friction_alpha), axis=0)

            cur_cohesion = np.repeat(0., cur_geo_cfg['num_particles']).astype(np.float32)
            self.init_cohesion = cur_cohesion if self.init_cohesion is None else np.concatenate((self.init_cohesion, cur_cohesion), axis=0)

    def clear_grads(self):
        self.simulator.clear_grads()

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
            self.simulator.C[p, 0] = ti.Matrix.zero(self.dtype, 3, 3)
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
        # self.device = self.init_particles.device
        self.num_particles[None] = self.init_particles.shape[0]
        self.dx[None], self.inv_dx[None] = 0.02, 50
        self.simulator.p_vol[None] = (self.dx[None] * 0.5) ** 3
        self.simulator.cached_states.clear()
        self.from_torch(self.init_particles,
                        self.init_velocities,
                        self.init_rhos,
                        self.init_mu,
                        self.init_lam,
                        self.init_yield_stress,
                        self.init_plastic_viscosity,
                        self.init_friction_alpha,
                        self.init_cohesion,
                        self.material)
        self.compute_particle_mass()
        self.simulator.cfl_satisfy[None] = True

    def rotate_translate_points(self, particles, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z):  # TODO
        pcd_res = o3d.geometry.PointCloud()
        pcd_res.points = o3d.utility.Vector3dVector(particles)

        R = pcd_res.get_rotation_matrix_from_xyz((0, rot_y, 0))
        mesh_r = copy.deepcopy(pcd_res)
        mesh_r.rotate(R, center=(0, 0, 0))

        R1 = mesh_r.get_rotation_matrix_from_xyz((0, 0, rot_z))
        mesh_r2 = copy.deepcopy(mesh_r)
        mesh_r2.rotate(R1, center=(0, 0, 0))

        R2 = mesh_r2.get_rotation_matrix_from_xyz((rot_x, 0, 0))
        mesh_r3 = copy.deepcopy(mesh_r2)
        mesh_r3.rotate(R2, center=(0, 0, 0)).translate((pos_x, pos_y, pos_z))

        final_transformed_points = np.asarray(mesh_r3.points, dtype=np.float32)
        return final_transformed_points

if __name__=='__main__':
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=0.9)