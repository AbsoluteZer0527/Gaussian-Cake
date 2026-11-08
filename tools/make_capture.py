#!/usr/bin/env python3
"""Make a 3D Gaussian Splatting .ply (the original 3DGS layout) from a mesh.

The splats sit on the mesh surface, and their colour is view-dependent: a
diffuse base colour plus a specular highlight, stored in degree-3 spherical
harmonics like a real capture. Useful for trying the .ply / SH code without
downloading a multi-hundred-MB capture.

    python3 tools/make_capture.py meshes/bunny.obj meshes/bunny_capture.ply --n 60000 --color 0.85,0.55,0.3
"""
import argparse
import numpy as np

C0 = 0.28209479177387814


def sh_basis(d):
    """Y_0..Y_15 at unit directions d (N x 3), 3DGS constants and signs."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    C1 = 0.4886025119029199
    C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
    C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
          -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
    return np.stack([
        np.full_like(x, C0),
        -C1 * y, C1 * z, -C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy),
        C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
        C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
        C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy),
    ], axis=1)


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append([float(v) for v in parts[1:4]])
            elif parts[0] == 'f':
                idx = [int(p.split('/')[0]) for p in parts[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):          # fan-triangulate polygons
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def quat_from_z(n):
    """Quaternions (w, x, y, z) of rotations taking +z to each unit normal."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(np.broadcast_to(z, n.shape), n)
    w = 1.0 + n[:, 2]
    q = np.concatenate([w[:, None], axis], axis=1)
    flip = w < 1e-6                                     # normal points straight down
    q[flip] = [0.0, 1.0, 0.0, 0.0]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh')
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=60000)
    ap.add_argument('--color', default='0.8,0.5,0.3')
    ap.add_argument('--shine', type=float, default=24.0, help='specular exponent')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    base = np.array([float(c) for c in a.color.split(',')])

    V, F = load_obj(a.mesh)
    tri = V[F]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = area > 1e-12
    tri, cross, area = tri[keep], cross[keep], area[keep]
    pick = rng.choice(len(tri), size=a.n, p=area / area.sum())
    u, v = rng.random(a.n), rng.random(a.n)
    out = u + v > 1
    u[out], v[out] = 1 - u[out], 1 - v[out]
    t = tri[pick]
    pos = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    nrm = cross[pick] / np.linalg.norm(cross[pick], axis=1, keepdims=True)

    # Outward normals: flip the ones pointing toward the centre (good enough
    # for the blobby test meshes this is meant for)
    centre = pos.mean(axis=0)
    nrm[np.einsum('ij,ij->i', nrm, pos - centre) < 0] *= -1

    # Colour as a function of the 3DGS view direction d (camera -> splat):
    # Lambert from a fixed light plus a Phong highlight seen along -d.
    dirs = rng.normal(size=(400, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Y = sh_basis(dirs)                                  # S x 16
    L = np.array([0.4, 0.8, 0.45]); L /= np.linalg.norm(L)
    ndl = np.clip(nrm @ L, 0.0, 1.0)
    refl = 2 * ndl[:, None] * nrm - L                   # light reflected about n
    view = -dirs                                        # splat -> camera
    spec = np.clip(refl @ view.T, 0.0, 1.0) ** a.shine  # N x S
    facing = (nrm @ view.T) > 0                         # no highlight from behind
    spec *= facing
    diffuse = (0.35 + 0.65 * ndl)[:, None] * base[None, :]
    coeffs = np.empty((a.n, 16, 3))
    for ch in range(3):
        f = diffuse[:, ch:ch + 1] + 0.9 * spec          # N x S
        coeffs[:, :, ch] = np.linalg.lstsq(Y, f.T, rcond=None)[0].T

    # 3DGS stores pre-activation values
    spacing = np.sqrt(area.sum() / a.n)
    scale = np.log(np.stack([np.full(a.n, 0.9 * spacing), np.full(a.n, 0.9 * spacing),
                             np.full(a.n, 0.1 * spacing)], axis=1))
    opacity = np.full(a.n, 3.0)                         # sigmoid -> 0.95
    f_dc = coeffs[:, 0, :] - 0.5 / C0                   # colour = 0.5 + C0 * f_dc
    f_rest = coeffs[:, 1:, :].transpose(0, 2, 1).reshape(a.n, 45)   # channel-major
    rot = quat_from_z(nrm)

    cols = np.concatenate([pos, nrm, f_dc, f_rest, opacity[:, None], scale, rot], axis=1).astype('<f4')
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] +
             [f'f_rest_{i}' for i in range(45)] +
             ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])
    with open(a.out, 'wb') as fp:
        header = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % a.n
        header += ''.join('property float %s\n' % n for n in names) + 'end_header\n'
        fp.write(header.encode())
        fp.write(cols.tobytes())
    print(f'wrote {a.out}: {a.n} splats, SH degree 3')


if __name__ == '__main__':
    main()
