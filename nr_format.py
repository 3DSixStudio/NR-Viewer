# -*- coding: utf-8 -*-
"""
Doc file .nr cua Ninja Ripper 2.x (header "NRIP", version 3).

Cau truc (little-endian), da phan tich tu file that:
    header : char[4] "NRIP" | u32 version | u32 0 | u32 0
    chunk  : u32 size (tinh ca 4 byte size) | char[4] tag | u32 id | data...

    TXTR : u32 count, {u32 slot, u32 unk, cstr filename} * count
    SHD1 : u32 count, {cstr hash} * count
    VATR : u32 count, {cstr semantic, u32 semIndex, u32 format, u32 offset, u32 comps} * count
    INDX : u32 count, u32 format (1 = u32, 0 = u16), indices...
    VERT : u32 count, u32 stride, raw vertex buffer
    GEOM : u32 flags (5 = input / 9 = output), u32 nRefs, u32 drawIndex,
           u32 instanceCount, u32, u32, u64 ptr, {char[4] tag, u32 id} * nRefs

    id 0 cua VATR/INDX/VERT = du lieu dau vao Vertex Shader (khong gian object).
    id >= 1 = dau ra Vertex Shader (SV_Position = clip space), moi instance 1 bo
    hoac gop chung trong 1 chunk.

    format attribute: 0 = float32, 3 = uint8 normalized, 5 = float16
"""
import os
import struct

import numpy as np

ATTR_FMT = {
    0: (np.float32, 4, False),
    1: (np.int32, 4, False),
    2: (np.uint32, 4, False),
    3: (np.uint8, 1, True),
    4: (np.uint16, 2, True),
    5: (np.float16, 2, False),
    6: (np.int16, 2, True),
    7: (np.int8, 1, True),
}


class NRAttr:
    __slots__ = ("name", "index", "fmt", "offset", "comps")

    def __init__(self, name, index, fmt, offset, comps):
        self.name, self.index, self.fmt, self.offset, self.comps = name, index, fmt, offset, comps

    @property
    def label(self):
        return f"{self.name}{self.index}"

    @property
    def size(self):
        return ATTR_FMT.get(self.fmt, (None, 4, False))[1] * self.comps

    def __repr__(self):
        return f"<{self.label} fmt={self.fmt} off={self.offset} x{self.comps}>"


class NRBuffer:
    """Mot bo VATR + INDX + VERT."""

    def __init__(self):
        self.attrs = []
        self.indices = None      # np.uint32 (N,)
        self.vcount = 0
        self.stride = 0
        self.index_count = 0
        self.raw = None          # bytes cua vertex buffer

    def read_attr(self, attr):
        dt, sz, norm = ATTR_FMT.get(attr.fmt, (None, 4, False))
        if dt is None or self.raw is None or self.vcount == 0:
            return None
        arr = np.frombuffer(self.raw, dtype=np.uint8).reshape(self.vcount, self.stride)
        block = arr[:, attr.offset:attr.offset + sz * attr.comps].copy()
        out = block.view(dt).reshape(self.vcount, attr.comps).astype(np.float32)
        if norm:
            out /= float(np.iinfo(dt).max)
        return out

    def find(self, pred):
        for a in self.attrs:
            if pred(a):
                return a
        return None


def _cstr(b, p):
    e = b.index(b"\0", p)
    return b[p:e].decode("latin1"), e + 1


class NRMesh:
    def __init__(self, path, header_only=False):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.textures = []        # [(slot, filename)]
        self.shaders = []
        self.buffers = {}         # id -> NRBuffer
        self.draw_index = -1
        self.instances = 1
        self._parse(header_only)

    # ------------------------------------------------------------------ parse
    def _parse(self, header_only):
        with open(self.path, "rb") as f:
            head = f.read(16)
            if head[:4] != b"NRIP":
                raise ValueError("Khong phai file Ninja Ripper (.nr)")
            f.seek(0, 2)
            n = f.tell()
            p = 16
            while p + 12 <= n:
                f.seek(p)
                hdr = f.read(12)
                size, = struct.unpack_from("<I", hdr, 0)
                if size < 12 or p + size > n:
                    break
                tag = hdr[4:8].decode("latin1")
                cid, = struct.unpack_from("<I", hdr, 8)
                big = tag in ("VERT", "INDX")
                b = f.read(8 if (big and header_only) else size - 12)
                self._chunk(tag, cid, b, header_only)
                p += size

    def _chunk(self, tag, cid, b, header_only):
        buf = self.buffers.setdefault(cid, NRBuffer()) if tag in ("VATR", "INDX", "VERT") else None
        if tag == "TXTR":
            cnt, = struct.unpack_from("<I", b, 0)
            q = 4
            for _ in range(cnt):
                slot, _unk = struct.unpack_from("<2I", b, q)
                q += 8
                s, q = _cstr(b, q)
                self.textures.append((slot, s))
        elif tag == "SHD1":
            cnt, = struct.unpack_from("<I", b, 0)
            q = 4
            for _ in range(cnt):
                s, q = _cstr(b, q)
                self.shaders.append(s)
        elif tag == "VATR":
            cnt, = struct.unpack_from("<I", b, 0)
            q = 4
            for _ in range(cnt):
                s, q = _cstr(b, q)
                idx, fmt, off, comps = struct.unpack_from("<4I", b, q)
                q += 16
                buf.attrs.append(NRAttr(s, idx, fmt, off, comps))
        elif tag == "INDX":
            cnt, fmt = struct.unpack_from("<2I", b, 0)
            buf.index_count = cnt
            if not header_only:
                dt = np.uint32 if fmt == 1 else np.uint16
                buf.indices = np.frombuffer(b, dtype=dt, count=cnt, offset=8).astype(np.uint32)
        elif tag == "VERT":
            cnt, stride = struct.unpack_from("<2I", b, 0)
            buf.vcount, buf.stride = cnt, stride
            if not header_only:
                buf.raw = b[8:8 + cnt * stride]
        elif tag == "GEOM":
            flags, nref, draw, inst = struct.unpack_from("<4I", b, 0)
            self.draw_index, self.instances = draw, max(1, inst)

    # --------------------------------------------------------------- helpers
    @property
    def input(self):
        return self.buffers.get(0)

    @property
    def outputs(self):
        return [self.buffers[k] for k in sorted(self.buffers) if k > 0]

    def vert_count(self):
        b = self.input
        return b.vcount if b else 0

    def tri_count(self):
        b = self.input
        return (b.index_count // 3) if b else 0

    def is_screen_quad(self):
        b = self.input
        return b is not None and b.vcount <= 4

    def position_attr(self):
        b = self.input
        if not b:
            return None
        a = b.find(lambda a: a.name.upper() == "POSITION")
        if a is None:
            a = b.find(lambda a: a.fmt == 0 and a.comps >= 3)
        if a is None:
            a = b.find(lambda a: a.fmt == 0 and a.comps >= 2)
        return a

    def uv_candidates(self):
        b = self.input
        if not b:
            return []
        return [a for a in b.attrs if a.comps == 2 and a.fmt in (0, 5)]

    def texture_path(self, slot_index=0):
        if not self.textures:
            return None
        slot_index = max(0, min(slot_index, len(self.textures) - 1))
        p = os.path.join(os.path.dirname(self.path), self.textures[slot_index][1])
        return p if os.path.isfile(p) else None

    # ---------------------------------------------------------------- export
    def local_geometry(self, uv_attr=None, with_color=False, color_label=None):
        """(positions Nx3, uvs Nx2|None, indices Mx3[, colors Nx3]) trong khong gian object."""
        b = self.input
        pa = self.position_attr()
        if b is None or pa is None or b.indices is None:
            return None
        pos = b.read_attr(pa)
        if pos.shape[1] == 2:
            pos = np.hstack([pos, np.zeros((len(pos), 1), np.float32)])
        pos = pos[:, :3]
        uvs = None
        if uv_attr is None:
            c = self.uv_candidates()
            uv_attr = c[0] if c else None
        if uv_attr is not None:
            uvs = b.read_attr(uv_attr)
        idx = _clean_indices(b.indices, len(pos))
        if with_color:
            cols = mesh_vertex_colors(self, color_label)
            if cols is not None:
                cols = cols[:len(pos)]
            return pos, uvs, idx, cols
        return pos, uvs, idx

    def view_geometry(self, fov_deg=60.0, aspect=16 / 9, uv_attr_label=None, clip_fn=None,
                      with_color=False, color_label=None):
        """Dung lai toa do camera tu SV_Position cua cac output (moi instance)."""
        outs = self.outputs
        if not outs:
            return None
        t = np.tan(np.radians(fov_deg) / 2.0)
        P, I, U, C = [], [], [], []
        base = 0
        in_uv = None
        if uv_attr_label is not None and self.input is not None:
            in_uv = self.input.find(lambda a: a.label == uv_attr_label)
        in_col = None
        if with_color:
            cands = _color_attrs(self.input)
            if color_label:
                in_col = next((x for x in cands if x.label == color_label), None)
            if in_col is None and cands:
                in_col = cands[0]
        for ob in outs:
            if ob.raw is None or ob.indices is None:
                continue
            sa = ob.find(lambda a: a.name.upper() == "SV_POSITION") or (ob.attrs[0] if ob.attrs else None)
            if sa is None or sa.comps < 4:
                continue
            c = ob.read_attr(sa)
            if clip_fn is not None:
                pos = clip_fn(c)          # vd: dung ma tran camera that tu file log
            else:
                w = c[:, 3]
                x = c[:, 0] * t * aspect
                y = c[:, 1] * t
                pos = np.stack([x, y, -w], axis=1).astype(np.float32)
            P.append(pos)
            uv = None
            # UV: lay tu input neu so dinh khop (instance lap lai input)
            ib = self.input
            if in_uv is not None and ib is not None and ib.vcount and ob.vcount % ib.vcount == 0:
                uv = np.tile(ib.read_attr(in_uv), (ob.vcount // ib.vcount, 1))
            U.append(uv if uv is not None else np.zeros((ob.vcount, 2), np.float32))
            if with_color:
                col = None
                if in_col is not None and ib is not None and ib.vcount and ob.vcount % ib.vcount == 0:
                    col = np.tile(ib.read_attr(in_col)[:, :3], (ob.vcount // ib.vcount, 1))
                C.append(col if col is not None else np.ones((ob.vcount, 3), np.float32))
            I.append(_clean_indices(ob.indices, len(pos)) + base)
            base += len(pos)
        if not P:
            return None
        if with_color:
            return np.vstack(P), np.vstack(U), np.vstack(I), (np.vstack(C) if C else None)
        return np.vstack(P), np.vstack(U), np.vstack(I)


def _clean_indices(idx, nverts):
    n = (len(idx) // 3) * 3
    tri = idx[:n].reshape(-1, 3)
    ok = (tri < nverts).all(axis=1)
    tri = tri[ok]
    # bo tam giac suy bien
    deg = (tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])
    return tri[~deg].astype(np.uint32)


def read_ripdesc(folder):
    """Doc file ripdesc -> dict thuoc tinh (executable, width, height...)."""
    path = os.path.join(folder, "ripdesc")
    info = {}
    if not os.path.isfile(path):
        return info
    d = open(path, "rb").read()
    i = d.find(b"PROP", 0x18)
    # bang thuoc tinh: cstr name, u16 type, u16 len, data
    p = d.find(b"PROP", 0x18) + 4 if i >= 0 else -1
    if p <= 0:
        return info
    p += 4  # bo count
    while p < len(d):
        try:
            e = d.index(b"\0", p)
        except ValueError:
            break
        name = d[p:e].decode("latin1")
        if not name or not name.isidentifier():
            break
        typ, ln = struct.unpack_from("<HH", d, e + 1)
        val = d[e + 5:e + 5 + ln]
        info[name] = struct.unpack("<I", val)[0] if typ == 1 and ln == 4 else val.decode("latin1", "replace")
        p = e + 5 + ln
    return info


def export_obj(mesh, out_path, uv_attr=None, flip_v=True):
    g = mesh.local_geometry(uv_attr)
    if g is None:
        raise ValueError("Mesh khong co du lieu hinh hoc")
    pos, uvs, tri = g
    tex = mesh.texture_path(0)
    base = os.path.splitext(out_path)[0]
    mtl_name = os.path.basename(base) + ".mtl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Exported by NR Viewer from {os.path.basename(mesh.path)}\n")
        f.write(f"mtllib {mtl_name}\no {mesh.name}\n")
        f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in pos))
        if uvs is not None:
            vv = (1.0 - uvs[:, 1]) if flip_v else uvs[:, 1]
            f.write("".join(f"vt {u:.6f} {v:.6f}\n" for u, v in zip(uvs[:, 0], vv)))
        f.write(f"usemtl {mesh.name}\n")
        if uvs is not None:
            f.write("".join(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n" for a, b, c in tri))
        else:
            f.write("".join(f"f {a+1} {b+1} {c+1}\n" for a, b, c in tri))
    with open(base + ".mtl", "w", encoding="utf-8") as f:
        f.write(f"newmtl {mesh.name}\nKd 1 1 1\n")
        if tex:
            f.write(f"map_Kd {tex}\n")


def export_scene_obj(items, out_path, flip_v=True):
    """items: list of (name, pos Nx3, uvs Nx2|None, tri Mx3, tex_path|None)."""
    base = os.path.splitext(out_path)[0]
    mtl_name = os.path.basename(base) + ".mtl"
    off = 0
    with open(out_path, "w", encoding="utf-8") as f, open(base + ".mtl", "w", encoding="utf-8") as fm:
        f.write(f"# Exported by NR Viewer\nmtllib {mtl_name}\n")
        for name, pos, uvs, tri, tex in items:
            f.write(f"o {name}\n")
            f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in pos))
            if uvs is None:
                uvs = np.zeros((len(pos), 2), np.float32)
            vv = (1.0 - uvs[:, 1]) if flip_v else uvs[:, 1]
            f.write("".join(f"vt {u:.6f} {v:.6f}\n" for u, v in zip(uvs[:, 0], vv)))
            f.write(f"usemtl {name}\n")
            t = tri + off + 1
            f.write("".join(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in t))
            off += len(pos)
            fm.write(f"newmtl {name}\nKd 1 1 1\n")
            if tex:
                fm.write(f"map_Kd {tex}\n")
            fm.write("\n")


def calibrate_projection(meshes, max_meshes=40):
    """
    Tu do FOV doc + ti le khung hinh cua camera game tu chinh du lieu mesh.
    Y tuong: voi mesh cung (rigid), anh xa object space -> camera space phai la
    xoay * scale theo truc (cac cot truc giao). Chon (sx, sy) sao cho dieu nay dung nhat.
    Tra ve (fov_deg, aspect, sai_so, so_mesh_dung) hoac None.
    """
    data = []
    for m in meshes:
        if len(data) >= max_meshes:
            break
        try:
            if m.is_screen_quad() or not m.outputs:
                continue
            ob = m.outputs[0]
            if ob.raw is None or not ob.attrs:
                continue
            g = m.local_geometry()
            if g is None:
                continue
            P = g[0].astype(np.float64)
            clip = ob.read_attr(ob.attrs[0]).astype(np.float64)
            if clip.shape[1] < 4 or len(P) != len(clip) or len(P) < 8:
                continue
            if (np.ptp(P, 0) < 1e-4).any():
                continue
            X = np.hstack([P, np.ones((len(P), 1))])
            Y = clip[:, [0, 1, 3]]
            A, *_ = np.linalg.lstsq(X, Y, rcond=None)
            if np.abs(X @ A - Y).max() > 1e-3 * (np.abs(Y).max() + 1e-9):
                continue  # mesh co skin / bien dang -> bo
            data.append(A[:3].T)
        except Exception:  # noqa
            continue
    if len(data) < 3:
        return None
    Ls = np.array(data)  # (n, 3, 3)

    def score(sx, sy):
        D = np.array([1 / sx, 1 / sy, 1.0])
        Lc = Ls * D[None, :, None]
        G = np.einsum("nki,nkj->nij", Lc, Lc)
        dg = np.sqrt(np.einsum("nii->ni", G)) + 1e-12
        C = G / (dg[:, :, None] * dg[:, None, :])
        off = np.abs(C - np.eye(3)[None]).max(axis=(1, 2))
        return float(np.median(off))

    best = min((score(sx, sy), sx, sy) for sx in np.geomspace(0.1, 10, 60) for sy in np.geomspace(0.1, 10, 60))
    for span in (0.15, 0.03, 0.006):
        _, bx, by = best
        best = min((score(sx, sy), sx, sy)
                   for sx in np.linspace(bx * (1 - span), bx * (1 + span), 25)
                   for sy in np.linspace(by * (1 - span), by * (1 + span), 25))
    err, sx, sy = best
    fov = float(np.degrees(2 * np.arctan(1.0 / sy)))
    return fov, float(sy / sx), err, len(data)


# ----------------------------------------------------------------------------- skybox
def _boundary_edges(tri):
    """Canh bien cua 1 nhom tam giac (canh chi thuoc 1 tam giac)."""
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    es = np.sort(e, axis=1)
    _, inv, cnt = np.unique(es, axis=0, return_inverse=True, return_counts=True)
    return e[cnt[inv.ravel()] == 1]


def _cluster_dirs(vecs, max_k, tol_deg=12.0):
    """Gom vector theo huong (khong ke dau). Tra ve [(huong don vi, tong do dai, so luong)]."""
    groups = []
    cos_t = np.cos(np.radians(tol_deg))
    order = np.argsort(-np.linalg.norm(vecs, axis=1))
    for v in vecs[order]:
        L = np.linalg.norm(v)
        if L < 1e-9:
            continue
        u = v / L
        for g in groups:
            c = float(u @ g[0])
            if abs(c) >= cos_t:
                if c < 0:
                    u = -u
                g[1] += u * L
                g[2] += L
                g[3] += 1
                g[0] = g[1] / np.linalg.norm(g[1])
                break
        else:
            groups.append([u, u * L, L, 1])
    groups.sort(key=lambda g: -g[2])
    return groups[:max_k]


def _planes(pos, tri, tol_deg=8.0):
    """Tach tam giac thanh cac mat phang (theo phap tuyen)."""
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1)
    ok = area > 1e-12
    tri, n = tri[ok], n[ok] / area[ok, None]
    labels = -np.ones(len(tri), int)
    reps = []
    cos_t = np.cos(np.radians(tol_deg))
    for i in np.argsort(-area[ok]):
        for k, r in enumerate(reps):
            if abs(n[i] @ r) >= cos_t:
                labels[i] = k
                break
        else:
            reps.append(n[i])
            labels[i] = len(reps) - 1
    return [tri[labels == k] for k in range(len(reps))]


def skybox_from_faces(parts, align=True):
    """
    Nan canh dua tren cac MAT skybox (chi can 2-3 mat ke nhau la du).
    parts: list (pos Nx3 trong canh dung lai, tri Mx3). Moi mat skybox that la hinh vuong,
    cac mat vuong goc nhau -> 3 huong canh a, b, c phai thanh 3 truc vuong goc, dai bang nhau.
    Tim ma tran M (3x3, quanh camera) bien a, b, c thanh 3 truc X/Y/Z cung do dai.
    """
    edges = []
    nfaces = 0
    for pos, tri in parts:
        pos = np.asarray(pos, np.float64)
        for ftri in _planes(pos, np.asarray(tri)):
            be = _boundary_edges(ftri)
            if len(be) < 3:
                continue
            vec = pos[be[:, 1]] - pos[be[:, 0]]
            two = _cluster_dirs(vec, 2)
            if len(two) < 2:
                continue
            nfaces += 1
            for g in two:
                # do dai canh mat ~ nua tong chieu dai bien theo huong do
                edges.append(g[0] * (g[2] / 2.0))
    if nfaces < 2:
        raise ValueError("Cần ít nhất 2 mặt skybox kề nhau (đang có %d)" % nfaces)
    dirs = _cluster_dirs(np.array(edges), 3, tol_deg=15.0)
    if len(dirs) < 3:
        raise ValueError("Các mặt chọn không đủ 3 hướng cạnh (cần 2 mặt KỀ nhau)")
    E = np.stack([g[0] * (g[2] / g[3]) for g in dirs], axis=1)   # cot = a, b, c (co do dai)
    # gan truc: huong gan "len" nhat (Y camera) -> Y
    up = np.array([0.0, 1.0, 0.0])
    iy = int(np.argmax([abs(E[:, k] @ up) / np.linalg.norm(E[:, k]) for k in range(3)]))
    rest = [k for k in range(3) if k != iy]
    E = E[:, [rest[0], iy, rest[1]]]
    if E[:, 1] @ up < 0:
        E[:, 1] *= -1
    # Z ve phia camera (camera nhin -Z)
    if E[:, 2] @ np.array([0, 0, 1.0]) < 0:
        E[:, 2] *= -1
    if np.linalg.det(E) < 0:
        E[:, 0] *= -1
    s = abs(np.linalg.det(E)) ** (1 / 3)
    Malign = s * np.linalg.inv(E)                  # a,b,c -> s*X, s*Y, s*Z
    U, S, Vt = np.linalg.svd(Malign)
    Q = U @ Vt
    M = Malign if align else Q.T @ Malign          # khong xoay: chi bo phan meo
    # sai so: do lech vuong goc / bang nhau sau khi nan
    F = Malign @ E
    Fn = F / np.linalg.norm(F, axis=0)
    ortho = float(np.abs(Fn.T @ Fn - np.eye(3)).max())
    ln = np.linalg.norm(F, axis=0)
    # do meo truoc khi nan: ti le gia tri ky di cua E (1,1,1 = vuong sach)
    sv = np.linalg.svd(E / np.linalg.norm(E, axis=0).mean(), compute_uv=False)
    info = {"faces": nfaces, "stretch": (sv / sv.max()).tolist(), "resid": ortho,
            "mirror": False, "size": float(ln.mean())}
    return M, info


def skybox_score(mesh, fov, aspect, flip_y):
    """Mesh nho, phang, o rat xa -> co the la 1 mat skybox. Tra ve khoang cach (0 = khong phai)."""
    try:
        if mesh.tri_count() < 2 or mesh.tri_count() > 200 or mesh.is_screen_quad():
            return 0.0
        g = mesh.view_geometry(fov, aspect)
        if g is None:
            return 0.0
        pos = g[0].astype(np.float64)
        if len(_planes(pos, g[2])) > 6:
            return 0.0
        return float(np.median(np.linalg.norm(pos, axis=1)))
    except Exception:  # noqa
        return 0.0


# ----------------------------------------------------------------------------- vertex color
def png_size(path):
    """Doc (width, height) tu header PNG, khong can thu vien ngoai. None neu khong doc duoc."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
            return None
        return struct.unpack(">II", head[16:24])
    except Exception:  # noqa
        return None


def _color_attrs(buf):
    """Cac thuoc tinh co the la vertex color: uint8-normalized, 3-4 kenh."""
    if buf is None:
        return []
    return [a for a in buf.attrs if a.fmt == 3 and a.comps in (3, 4)]


class _NRColorMixin:
    pass


def mesh_color_attrs(mesh):
    return _color_attrs(mesh.input)


def mesh_vertex_colors(mesh, attr_label=None):
    """(N,3) float 0..1 hoac None. Mac dinh lay kenh color dau tien."""
    b = mesh.input
    cands = _color_attrs(b)
    if not cands:
        return None
    a = None
    if attr_label:
        a = next((x for x in cands if x.label == attr_label), None)
    if a is None:
        a = cands[0]
    col = b.read_attr(a)
    return col[:, :3] if col is not None else None


def split_faces_by_color(colors01, tri):
    """
    colors01: (N,3) 0..1. Tra ve (palette [(r,g,b)...], face_mat (M,) int) - mau y het (exact).
    Mau moi tam giac = lam tron trung binh 3 dinh ve uint8.
    """
    q = np.clip(np.round(np.asarray(colors01)[:, :3] * 255.0), 0, 255).astype(np.int32)
    tri = np.asarray(tri)
    face = q[tri]                                   # (M,3,3)
    fc = np.round(face.mean(axis=1)).astype(np.int32)   # (M,3)
    uniq, inv = np.unique(fc, axis=0, return_inverse=True)
    palette = (uniq.astype(np.float32) / 255.0)
    return [tuple(c) for c in palette], inv.astype(np.int32).ravel()
