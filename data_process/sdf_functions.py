import numpy as np
from sdf import mesh
from sdf import *
# import open3d as o3d
import os
import torch

def vector_length(p):
    return np.linalg.norm(p, axis=-1)

def vector_normalize(p):
    return p / np.linalg.norm(p)

def vector_stack(*arrs):
    return np.stack(arrs, axis=-1)

# def rounded_box(size, radius):
#     size = np.array(size)
#     def f(p):
#         q = np.abs(p) - size / 2 + radius
#         return vector_length(np.maximum(q, 0)) + np.minimum(np.amax(q, axis=1), 0) - radius
#     return f

# def sphere(radius=1, center=((0, 0, 0))):
#     def sdf_func(p):
#         return vector_length(p - center) - radius
#     return sdf_func

# def box(size=1, center=((0, 0, 0)), a=None, b=None):
#     if a is not None and b is not None:
#         a = np.array(a)
#         b = np.array(b)
#         size = b - a
#         center = a + size / 2
#         return box(size, center)
#     size = np.array(size)
#     def f(p):
#         q = np.abs(p - center) - size / 2
#         return vector_length(np.maximum(q, 0)) + np.minimum(np.amax(q, axis=1), 0)
#     return f

def torus(r1, r2):
    def f(p):
        xy = p[:,[0,1]]
        z = p[:,2]
        a = vector_length(xy) - r1
        b = vector_length(vector_stack(a, z)) - r2
        return b
    return f

def blobby(s_r1, s_r2, f_scale=None):
    s = sphere(s_r1) #sphere(0.75)
    s = s.translate(Z * -3) | s.translate(Z * 3)
    s = s.union(capsule(Z * -3, Z * 3, 0.5), k=1)

    # f = sphere(1.5).union(s.orient(X), s.orient(Y), s.orient(Z), k=1)
    f = sphere(s_r2).union(s.orient(X), s.orient(Y), s.orient(Z), k=1)
    # f = f.scale(0.05)
    # f = f.scale(f_scale)
    return f

def toy():
    f = sphere(1) & box(1.5)
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    return f

def knurling():
    # main body
    f = rounded_cylinder(1, 0.1, 5)

    # knurling
    x = box((1, 1, 4)).rotate(pi / 4)
    x = x.circular_array(24, 1.6)
    x = x.twist(0.75) | x.twist(-0.75)
    f -= x.k(0.1)

    # central hole
    f -= cylinder(0.5).k(0.1)

    # vent holes
    c = cylinder(0.25).orient(X)
    f -= c.translate(Z * -2.5).k(0.1)
    f -= c.translate(Z * 2.5).k(0.1)
    return f

def weave():
    f = rounded_box([3.2, 1, 0.25], 0.1).translate((1.5, 0, 0.0625))
    f = f.bend_linear(X * 0.75, X * 2.25, Z * -0.1875, ease.in_out_quad)
    f = f.circular_array(2, 0)

    f = f.repeat((2.7, 5.4, 0), padding=1)
    f |= f.translate((2.7 / 2, 2.7, 0))

    f &= cylinder(10)
    f |= (cylinder(12) - cylinder(10)) & slab(z0=-0.5, z1=0.5).k(0.25)
    return f

def gearlike():
    f = sphere(2) & slab(z0=-0.5, z1=0.5).k(0.1)
    f -= cylinder(1).k(0.1)
    f -= cylinder(0.25).circular_array(16, 2).k(0.1)
    return f

def pawn():
    def section(z0, z1, d0, d1, e=ease.linear):
        f = cylinder(d0/2).transition_linear(
            cylinder(d1/2), Z * z0, Z * z1, e)
        return f & slab(z0=z0, z1=z1)

    f = section(0, 0.2, 1, 1.25)
    f |= section(0.2, 0.3, 1.25, 1).k(0.05)
    f |= rounded_cylinder(0.6, 0.1, 0.2).translate(Z * 0.4).k(0.05)
    f |= section(0.5, 1.75, 1, 0.25, ease.out_quad).k(0.01)
    f |= section(1.75, 1.85, 0.25, 0.5).k(0.01)
    f |= section(1.85, 1.90, 0.5, 0.25).k(0.05)
    f |= sphere(0.3).translate(Z * 2.15).k(0.05)

    return f

def generate_sdf_particles(geometry, *geo_params):
    pts = None
    if geometry == "torus":
        t = torus(r1=geo_params[0][0], r2=geo_params[0][1])
        pts = mesh.generate(t)
    elif geometry == "sphere":
        s = sphere(geo_params[0][0])
        pts = mesh.generate(s)
    elif geometry == "box":
        b = box((geo_params[0][0], geo_params[0][1], geo_params[0][2]))
        pts = mesh.generate(b)
    elif geometry == "rounded_box":
        rb = rounded_box(geo_params[0][0], geo_params[0][0])
        pts = mesh.generate(rb)
    elif geometry == "blobby":
        blob = blobby(s_r1=geo_params[0][0], s_r2=geo_params[0][1]).scale(0.03)
        pts = mesh.generate(blob)
    elif geometry == "pyramid":
        p = pyramid(geo_params[0][0]).scale(0.3)
        pts = mesh.generate(p)
    elif geometry == "rounded_cylinder":
        rcy = rounded_cylinder(geo_params[0][0], geo_params[0][1], geo_params[0][2]).scale(0.1)
        pts = mesh.generate(rcy)
    elif geometry == "ellipsoid":
        ellp = ellipsoid((geo_params[0][0], geo_params[0][1], geo_params[0][2])).scale(0.1)
        pts = mesh.generate(ellp)
    elif geometry == "rounded_cone":
        rc = rounded_cone(geo_params[0][0], geo_params[0][1], geo_params[0][2]).scale(0.1)
        pts = mesh.generate(rc)
    elif geometry == "capped_cone":
        cc = capped_cone(-Z, Z, geo_params[0][0], geo_params[0][1]).scale(0.1)
        pts = mesh.generate(cc)
    elif geometry == "octahedron":
        octh = octahedron(geo_params[0][0]).scale(0.1)
        pts = mesh.generate(octh)
    elif geometry == "dodecahedron":
        dodec = dodecahedron(geo_params[0][0]).scale(0.1)
        pts = mesh.generate(dodec)
    elif geometry == "toy":
        t = toy().scale(0.1)
        pts = mesh.generate(t)
    elif geometry == "knurling":
        k = knurling().scale(0.05)
        pts = mesh.generate(k)
    elif geometry == "weave":
        w = weave().scale(0.01)
        pts = mesh.generate(w)
    elif geometry == "gearlike":
        g = gearlike().scale(0.05)
        pts = mesh.generate(g)
    elif geometry == "pawn":
        p = pawn().scale(0.1)
        pts = mesh.generate(p)

    pts = torch.Tensor(pts)
    return pts

if __name__ == "__main__":
    # s = sphere(0.1)
    # pts = mesh.generate(s)
    # np.savez('sphere_01.npz', points=pts)
    # b = box(1.5)
    # pts = mesh.generate(b)
    # np.savez('box.npz', points=pts)

    t = torus(0.1, 0.025)
    pts = mesh.generate(t)
    # np.savez('torus_01_025.npz', points=pts)
    # pts = generate_sdf_particles("torus", [0.1, 0.025])

    rb = rounded_box(0.2, 0.02)
    pts2 = mesh.generate(rb)
    np.savez('rounded_box2.npz', points=pts2, torus = pts)



