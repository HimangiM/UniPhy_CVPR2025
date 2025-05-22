import torch
import os
from mpm_simulator_learnable_grad import MPMSimulatorLearnableGrad
import taichi as ti
# from constitutive_functions import elastic_constitutive_model, newtonian_constitutive_model, sand_nonnewtonian_plasticine
from einops.layers.torch import Rearrange

# get functions: taichi to torch
# update functions: torch to taichi

@ti.data_oriented
class MPMSimulatorConstitutive(MPMSimulatorLearnableGrad):
    def __init__(self, dtype, dt, frame_dt, particle_layout, dx, inv_dx, n_particles, gravity, material, cuda_chunk_size,
                 constitutive_function, stress_model, particles_ti_root, tb_writer, stress_gt, device, embed_dim, fproj_model, **kwargs):
        super().__init__(dtype, dt, frame_dt, particle_layout, dx, inv_dx, n_particles, gravity, material, cuda_chunk_size, **kwargs)

        self.constitutive_function = constitutive_function
        self.stress_model = stress_model
        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=3, d2=3)
        self.device = device
        self.Ftmp_substep = torch.zeros([particles_ti_root, cuda_chunk_size+1, 3, 3], dtype=torch.float32).to(self.device)
        self.U_substep = torch.zeros([particles_ti_root, cuda_chunk_size+1, 3, 3], dtype=torch.float32).to(self.device)
        self.V_substep = torch.zeros([particles_ti_root, cuda_chunk_size+1, 3, 3], dtype=torch.float32).to(self.device)
        self.sigma_substep = torch.zeros([particles_ti_root, cuda_chunk_size+1, 3, 3], dtype=torch.float32).to(self.device)
        self.F_substep = torch.zeros([particles_ti_root, cuda_chunk_size+1, 3, 3], dtype=torch.float32).to(self.device)
        self.C_substep = torch.zeros([particles_ti_root, cuda_chunk_size+1, 3, 3], dtype=torch.float32).to(self.device)
        self.stress_grad = torch.zeros([particles_ti_root, cuda_chunk_size+1, 3, 3], dtype=torch.float32).to(self.device)
        self.F_grad = torch.zeros([particles_ti_root, cuda_chunk_size+1, 3, 3], dtype=torch.float32).to(self.device)
        self.stress_gt = stress_gt
        self.tb_writer = tb_writer
        self.traj_latent = torch.zeros([1, embed_dim], dtype=torch.float32).to(self.device)
        self.traj_latent_grad = torch.zeros(embed_dim, dtype=torch.float32).to(self.device)
        self.embed_dim = embed_dim
        self.fproj_model = fproj_model
        self.prev_loss = 0.

    # F_to_stress function/Ftmp_to_Fproj function
    # latent: taichi to torch
    @ti.kernel
    def get_latent(self, latent_vec: ti.types.ndarray()):
        for e in range(self.embed_dim):
            latent_vec[0, e] = ti.cast(self.traj_latent_ti[e], ti.f32)

    @ti.kernel
    def update_traj_latent_grad(self, traj_latent_grad_np: ti.types.ndarray()):
        for e in range(self.embed_dim):
            self.traj_latent_ti.grad[e] = traj_latent_grad_np[e]

    @ti.kernel
    def get_traj_latent_grad(self, traj_latent_grad: ti.types.ndarray()):
        for e in range(self.embed_dim):
            traj_latent_grad[e] = ti.cast(self.traj_latent_ti.grad[e], ti.f32)

    @ti.kernel
    def get_stress_grad(self, step: ti.i32, stress_grad: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    stress_grad[p, step, i, j] = ti.cast(self.stress.grad[p, step][i, j], ti.f32)

    @ti.kernel
    def get_F_grad(self, step: ti.i32, F_grad: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    F_grad[p, step, i, j] = ti.cast(self.F.grad[p, step][i, j], ti.f32)

    @ti.kernel
    def get_Ftmp(self, step:ti.i32, F_tmp_t: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    F_tmp_t[p, step, i, j] = ti.cast(self.F_tmp[p, step][i, j], ti.f32)

    # F_to_stress function
    # F: taichi to torch
    @ti.kernel
    def get_F(self, step:ti.i32, F_t: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    F_t[p, step, i, j] = ti.cast(self.F[p, step][i, j], ti.f32)  # self.F = 3 x 3

    # F_to_stress function
    # C: taichi to torch
    @ti.kernel
    def get_C(self, step:ti.i32, C_t: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    C_t[p, step, i, j] = ti.cast(self.C[p, step][i, j], ti.f32)  # self.C = 3 x 3

    # F_to_stress function
    # U: taichi to torch
    @ti.kernel
    def get_U(self, step:ti.i32, U_t: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    U_t[p, step, i, j] = ti.cast(self.U[p, step][i, j], ti.f32)  # self.C = 3 x 3

    # F_to_stress function
    # V: taichi to torch
    @ti.kernel
    def get_V(self, step:ti.i32, V_t: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    V_t[p, step, i, j] = ti.cast(self.V[p, step][i, j], ti.f32)  # self.C = 3 x 3

    # F_to_stress function
    # sig: taichi to torch
    @ti.kernel
    def get_sig(self, step:ti.i32, sig_t: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    sig_t[p, step, i, j] = ti.cast(self.sig[p, step][i, j], ti.f32)  # self.C = 3 x 3

    @ti.kernel
    def update_Fproj_ti(self, particle_Fproj: ti.types.ndarray(), s: ti.i32):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    self.F[p, s][i, j] = particle_Fproj[p, i, j]

    # F_to_stress function
    # stress: torch to taichi
    @ti.kernel
    def update_stress_ti(self, particle_stress: ti.types.ndarray(), s: ti.i32):
        for p in range(self.n_particles[None]):
            for i_stress in ti.static(range(3)):
                for j_stress in ti.static(range(3)):
                    self.stress[p, s][i_stress, j_stress] = particle_stress[p, i_stress, j_stress]

    @ti.kernel
    def update_x_ti(self, particle_x: ti.types.ndarray(), s: ti.i32):
        for p in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                self.x[p, s][d] = particle_x[p, d]

    @ti.kernel
    def update_F_grad(self, F_grad_np: ti.types.ndarray(), s: ti.i32):
        for p in range(self.n_particles[None]):
            # self.F.grad[p, s] = F_grad_np[p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    self.F.grad[p, s][i, j] = F_grad_np[p, i, j]

    @ti.kernel
    def update_Ftmp_grad(self, F_tmp_grad_np: ti.types.ndarray(), s: ti.i32):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    self.F_tmp.grad[p, s][i, j] = F_tmp_grad_np[p, i, j]

    def Ftmp_to_Fproj(self, t, traj_id):
        self.get_Ftmp(t+1, self.Ftmp_substep)
        Ftmp_p_torch = self.Ftmp_substep[:self.n_particles[None], t+1, :, :]
        Ftmp_p_torch.requires_grad_(True)

        self.get_U(t+1, self.U_substep)
        U_p_torch = self.U_substep[:self.n_particles[None], t+1, :, :]
        U_p_torch.requires_grad_(True)

        self.get_V(t+1, self.V_substep)
        V_p_torch = self.V_substep[:self.n_particles[None], t+1, :, :]
        V_p_torch.requires_grad_(True)

        # U_p_torch[0] @ sigma_p_torch[0] @ V_p_torch[0].T
        self.get_latent(self.traj_latent)
        self.fproj_model.trajectory_latent.from_pretrained(self.traj_latent)
        Fproj_torch = self.fproj_model(Ftmp_p_torch, U_p_torch, V_p_torch, traj_id)
        # loss_mse = torch.nn.MSELoss()
        # cur_loss = loss_mse(Ftmp_p_torch, Fproj_torch)
        # print (cur_loss, cur_loss - self.prev_loss)
        # self.prev_loss = loss_mse(Ftmp_p_torch, Fproj_torch)

        if len((torch.isnan(Fproj_torch) == True).nonzero()) > 0 or len((torch.isnan(self.traj_latent) == True).nonzero()) > 0:
            import ipdb; ipdb.set_trace()

        self.update_Fproj_ti(Fproj_torch, t+1)

        # @ti.kernel
    def F_to_stress(self, t, traj_id):
        self.get_F(t+1, self.F_substep)
        F_p_torch = self.F_substep[:self.n_particles[None], t+1, :, :]
        F_p_torch.requires_grad_(True)

        self.get_C(t, self.C_substep)
        C_p_torch = self.C_substep[:self.n_particles[None], t, :, :]
        C_p_torch.requires_grad_(True)

        self.get_latent(self.traj_latent)
        self.stress_model.trajectory_latent.from_pretrained(self.traj_latent)
        stress_torch = self.stress_model(F_p_torch, C_p_torch, traj_id)
        # stress_torch = torch.clamp(stress_torch, -511906.5312, 358896.1562)

        # print ("Max of stress:", torch.max(stress_torch))
        if len((torch.isnan(stress_torch) == True).nonzero()) > 0 or len((torch.isnan(self.traj_latent) == True).nonzero()) > 0 or len((torch.isnan(F_p_torch) == True).nonzero()) > 0:
            print ("nan in stress_torch or traj_latent or F_p_torch")

        self.update_stress_ti(stress_torch, t+1) # this line gives the error of accessing .grad of leaf Tensor

    @ti.kernel
    def p2g_stress(self, s: ti.i32):
        ti.block_local(self.grid_m)
        ti.block_local(self.grid_v_in)
        ti.block_local(self.grid_v_out)
        for p in range(self.n_particles[None]):
            base = ti.floor(self.x[p, s] * self.inv_dx[None] - 0.5).cast(int)
            fx = self.x[p, s] * self.inv_dx[None] - base.cast(self.dtype)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

            stress_val = (-self.dt[None] * self.p_vol[None] * 4 * self.inv_dx[None]**2) * self.stress[p, s+1]
            affine = stress_val + self.p_mass[p] * self.C[p, s]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), self.dtype) - fx) * self.dx[None]
                        weight = w[i][0] * w[j][1] * w[k][2]
                        self.grid_v_in[base + offset] += \
                            weight * (self.p_mass[p] * self.v[p, s] + affine @ dpos)
                        self.grid_m[base + offset] += weight * self.p_mass[p]

    def advance_F_stress(self, t, traj_id):
        if self.cfl_satisfy[None]:
            self.substep_F_stress(t, traj_id, cache=True)

    def substep_F_stress(self, t, traj_id, cache=True):
        local_index = t % self.cuda_chunk_size

        self.grid.deactivate_all()
        self.compute_F_tmp(local_index)
        self.svd(local_index)

        self.Ftmp_to_Fproj(local_index, traj_id)

        self.F_to_stress(local_index, traj_id)
        self.p2g_stress(local_index)

        self.grid_op(local_index)
        self.g2p(local_index)
        self.check_cfl(local_index + 1)

        if (local_index == self.cuda_chunk_size-1) and cache:
            self.push_to_memory()

    def advance_gtF_stress(self, t, traj_id):
        if self.cfl_satisfy[None]:
            self.substep_gtF_stress(t, traj_id, cache=True)

    def substep_gtF_stress(self, t, traj_id, cache=True):
        local_index = t % self.cuda_chunk_size

        self.grid.deactivate_all()
        self.compute_F_tmp(local_index)
        self.svd(local_index)
        self.project_F(local_index)

        self.F_to_stress(local_index, traj_id)
        self.p2g_stress(local_index)

        self.grid_op(local_index)
        self.g2p(local_index)
        self.check_cfl(local_index + 1)

        if (local_index == self.cuda_chunk_size-1) and cache:
            self.push_to_memory()

    def advance_F_gtstress(self, t, traj_id):
        if self.cfl_satisfy[None]:
            self.substep_F_gtstress(t, traj_id, cache=True)

    def substep_F_gtstress(self, t, traj_id, cache=True):
        local_index = t % self.cuda_chunk_size

        self.grid.deactivate_all()
        self.compute_F_tmp(local_index)
        self.svd(local_index)

        self.Ftmp_to_Fproj(local_index, traj_id)

        self.p2g(local_index)
        self.grid_op(local_index)
        self.g2p(local_index)
        self.check_cfl(local_index + 1)

        if (local_index == self.cuda_chunk_size-1) and cache:
            self.push_to_memory()

    def Ftmp_to_Fproj_grad(self, t: ti.i32, traj_id):
        Ftmp_p_torch = self.Ftmp_substep[:self.n_particles[None], t+1, :, :]
        Ftmp_p_torch.requires_grad_(True)

        U_p_torch = self.U_substep[:self.n_particles[None], t+1, :, :]
        U_p_torch.requires_grad_(True)

        V_p_torch = self.V_substep[:self.n_particles[None], t+1, :, :]
        V_p_torch.requires_grad_(True)

        self.get_latent(self.traj_latent)
        self.fproj_model.trajectory_latent.from_pretrained(self.traj_latent)

        Fproj_torch = self.fproj_model(Ftmp_p_torch, U_p_torch, V_p_torch, traj_id)

        self.get_F_grad(t+1, self.F_grad)
        F_grad_cur_torch = self.F_grad[:self.n_particles[None], t+1, :, :]

        if len((torch.isnan(Fproj_torch) == True).nonzero()) > 0 or len((torch.isnan(self.traj_latent) == True).nonzero()) > 0 or len((torch.isnan(F_grad_cur_torch) == True).nonzero()):
            import ipdb; ipdb.set_trace()

        Fproj_torch.backward(F_grad_cur_torch)
        self.update_Ftmp_grad(Ftmp_p_torch.grad, t+1)

    # @ti.kernel
    def F_to_stress_grad(self, t:ti.i32, traj_id):
        # F_p = self.F
        # F_p_np = F_p.to_torch()[:self.n_particles[None]]
        # F_p_torch = torch.Tensor(F_p_np[:, t+1, :, :]).to(self.device)

        F_p_torch = self.F_substep[:self.n_particles[None], t+1, :, :]
        F_p_torch.requires_grad_(True)
        # F_p_torch.retain_grad()

        # traj_latent_torch = self.traj_latent
        self.get_latent(self.traj_latent)  # transfers latent from traj_latent_ti to self.traj_latent torch
        self.stress_model.trajectory_latent.from_pretrained(self.traj_latent)

        C_p_torch = self.C_substep[:self.n_particles[None], t, :, :]
        C_p_torch.requires_grad_(True)

        stress_torch = self.stress_model(F_p_torch, C_p_torch, traj_id)  # P x 3 x 3

        self.get_stress_grad(t+1, self.stress_grad)
        s_grad_cur_torch = self.stress_grad[:self.n_particles[None], t+1, :, :]

        if len((torch.isnan(s_grad_cur_torch) == True).nonzero()) > 0 or len((torch.isnan(self.traj_latent) == True).nonzero()) > 0 \
                or len((torch.isnan(F_p_torch) == True).nonzero()) > 0 or len((torch.isnan(s_grad_cur_torch) == True).nonzero()) > 0:  # nan and -inf in s_grad_cur_torch
            import ipdb; ipdb.set_trace()

        stress_torch.backward(gradient=s_grad_cur_torch)
        self.update_F_grad(F_p_torch.grad, t+1)
        # self.update_traj_latent_grad(traj_latent_torch.grad)
        # self.get_traj_latent_grad(self.traj_latent_grad)
        # traj_latent_torch.backward(gradient=traj_latent_torch.grad)

    def advance_grad_F_stress(self, f, traj_id):
        # import ipdb; ipdb.set_trace()
        for i in reversed(range(self.n_substeps[None] * f, self.n_substeps[None] * (f+1))):
            if self.cfl_satisfy[None]:
                self.substep_grad_F_stress(i, traj_id)

    def substep_grad_F_stress(self, s, traj_id):
        local_index = s % self.cuda_chunk_size
        if local_index == self.cuda_chunk_size-1:
            self.pop_from_memory()

        self.grid.deactivate_all()
        self.compute_F_tmp(local_index)
        self.svd(local_index)
        self.p2g_stress(local_index)
        self.grid_op(local_index)

        self.clear_svd_grads()
        self.g2p.grad(local_index)
        self.grid_op.grad(local_index)
        self.p2g_stress.grad(local_index)
        self.F_to_stress_grad(local_index, traj_id)
        self.Ftmp_to_Fproj_grad(local_index, traj_id)
        self.svd_grad(local_index)
        self.compute_F_tmp.grad(local_index)