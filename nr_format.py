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

# Khop CHINH XAC enum VaType cua Ninja Ripper 2 (giong importer Blender):
#   Float=0 Uint32=1 Uint16=2 Uint8=3 Sint32=4 Sint16=5 Sint8=6
# (dt, so byte, co chuan hoa /max khi doc chung). Doc mau dung ham rieng ben duoi.
ATTR_FMT = {
    0: (np.float32, 4, False),   # Float
    1: (np.uint32,  4, False),   # Uint32
    2: (np.uint16,  2, True),    # Uint16
    3: (np.uint8,   1, True),    # Uint8
    4: (np.int32,   4, False),   # Sint32
    5: (np.int16,   2, True),    # Sint16
    6: (np.int8,    1, True),    # Sint8
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


TOPO_TRI, TOPO_POINT, TOPO_LINE = 1, 2, 3      # nrfile.PrimitiveTopology
STAGE_PREVS = 1                                  # nrfile.ShaderStage.PreVs


class NRMesh:
    """Doc file .nr Y HET io_import_nr (nrfile.py + nrblendimp._importMeshImpl).

    - Moi chunk co 'idx' = so thu tu RIENG theo tung loai tag (VATR#0, VATR#1...).
    - Moi GEOM liet ke chinh xac cac chunk no dung qua cap (tag, idx).
    - Moi GEOM la 1 mesh. Blender (che do PostVS) tao 1 object cho MOI GEOM
      khong phai PreVs, dat ten = ten file (trung thi Blender tu them .001...).
    - De co toa do object-space, moi mesh Vs duoc ghep voi GEOM PreVs cung drawId.
    """

    def __init__(self, path, header_only=False, draw_sel=None):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.shaders = []
        self.buffers = {}         # geom index -> NRBuffer (da gom dung chunk cua GEOM do)
        self.geoms = []           # list dict thong tin GEOM
        self.draw_index = -1
        self.instances = 1
        self.draw_sel = draw_sel  # chon mesh thu k trong file
        self._ch = {}             # tag -> [du lieu chunk theo idx]
        self._parse(header_only)
        self._build_geoms()
        self._build_draws()

    # ------------------------------------------------------------ doc chunk
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
                big = tag in ("VERT", "INDX")
                b = f.read(8 if (big and header_only) else size - 12)
                self._ch.setdefault(tag, []).append(self._read_chunk(tag, b, header_only))
                p += size

    def _read_chunk(self, tag, b, header_only):
        if tag == "TXTR":
            out = []
            cnt, = struct.unpack_from("<I", b, 0)
            q = 4
            for _ in range(cnt):
                slot, _unk = struct.unpack_from("<2I", b, q)
                q += 8
                s, q = _cstr(b, q)
                out.append((slot, s))
            return out
        if tag == "SHD1":
            cnt, = struct.unpack_from("<I", b, 0)
            q = 4
            for _ in range(cnt):
                s, q = _cstr(b, q)
                self.shaders.append(s)
            return None
        if tag == "VATR":
            out = []
            cnt, = struct.unpack_from("<I", b, 0)
            q = 4
            for _ in range(cnt):
                s, q = _cstr(b, q)
                idx, fmt, off, comps = struct.unpack_from("<4I", b, q)
                q += 16
                out.append(NRAttr(s, idx, fmt, off, comps))
            return out
        if tag == "INDX":
            cnt, topo = struct.unpack_from("<2I", b, 0)     # INDX: count, topology; index LUON uint32
            ind = None
            if not header_only:
                ind = np.frombuffer(b, dtype=np.uint32, count=cnt, offset=8).copy()
            return (cnt, topo, ind)
        if tag == "VERT":
            cnt, stride = struct.unpack_from("<2I", b, 0)
            raw = None if header_only else b[8:8 + cnt * stride]
            return (cnt, stride, raw)
        if tag == "GEOM":
            flags, nref, draw, inst, rip, g0, g1, frame = struct.unpack_from("<8I", b, 0)
            refs = [struct.unpack_from("<2I", b, 32 + 8 * i) for i in range(nref)]
            refs = [(struct.pack("<I", t).decode("latin1"), i) for t, i in refs]
            return {"flags": flags, "topo": flags & 3, "stage": (flags & 0x1C) >> 2,
                    "draw": draw, "inst": max(1, inst), "refs": refs,
                    "group0": g0, "group1": g1, "frame": frame}
        return None

    def _ref(self, g, tag):
        """Chunk dau tien loai 'tag' ma GEOM g tham chieu (nhu Geometry.__getChunk / getX(0))."""
        for t, i in g["refs"]:
            if t == tag:
                arr = self._ch.get(tag) or []
                return arr[i] if i < len(arr) else None
        return None

    def _build_geoms(self):
        for gi, g in enumerate(self._ch.get("GEOM", [])):
            buf = NRBuffer()
            va = self._ref(g, "VATR")
            if va:
                buf.attrs = list(va)
            vt = self._ref(g, "VERT")
            if vt:
                buf.vcount, buf.stride, buf.raw = vt
            ix = self._ref(g, "INDX")
            if ix:
                cnt, _topo, ind = ix
                if g["topo"] == TOPO_LINE:              # LineList -> tam giac suy bien (a,b,b) de van co mat
                    buf.index_count = (cnt // 2) * 3
                    if ind is not None:
                        e = ind[:(len(ind) // 2) * 2].reshape(-1, 2)
                        buf.indices = np.stack([e[:, 0], e[:, 1], e[:, 1]], 1).reshape(-1)
                elif g["topo"] == TOPO_TRI:
                    buf.index_count = cnt
                    buf.indices = ind
            g["textures"] = list(self._ref(g, "TXTR") or [])
            self.buffers[gi] = buf
            self.geoms.append(g)

    def _valid_like_blender(self, gi, postvs):
        """Cac dieu kien 'continue' trong _importMeshImpl."""
        g, b = self.geoms[gi], self.buffers[gi]
        if not b.attrs or not b.vcount:
            return False
        if postvs and b.attrs[0].comps != 4:
            return False
        if g["topo"] == TOPO_TRI:
            return self._ref(g, "INDX") is not None and b.vcount >= 3
        if g["topo"] == TOPO_LINE:
            return self._ref(g, "INDX") is not None and b.vcount >= 2
        return g["topo"] == TOPO_POINT

    def _build_draws(self):
        """Danh sach mesh hien thi = moi GEOM khong phai PreVs (y het Blender PostVS),
        ghep voi GEOM PreVs cung drawId de lay toa do object-space.
        GEOM PreVs khong co cap Vs nao thi cung giu lai (khong bo mesh nao)."""
        pre_by_draw, pre_by_vatr = {}, {}
        for gi, g in enumerate(self.geoms):
            if g["stage"] == STAGE_PREVS and self._valid_like_blender(gi, False):
                pre_by_draw.setdefault(g["draw"], []).append(gi)
                v = [i for t, i in g["refs"] if t == "VATR"]
                if v:
                    pre_by_vatr.setdefault(v[0], gi)
        draws, used_pre = [], set()
        for gi, g in enumerate(self.geoms):
            if g["stage"] == STAGE_PREVS or not self._valid_like_blender(gi, True):
                continue
            # Ghep chinh xac: GEOM Vs tu tham chieu VATR cua PreVs o vi tri thu 2
            v = [i for t, i in g["refs"] if t == "VATR"]
            pre = pre_by_vatr.get(v[1]) if len(v) > 1 else None
            if pre is None:                       # du phong: cung drawId, uu tien cung so dinh
                cands = pre_by_draw.get(g["draw"], [])
                vc = self.buffers[gi].vcount
                pre = next((c for c in cands if self.buffers[c].vcount == vc), cands[0] if cands else None)
            if pre is not None:
                used_pre.add(pre)
            draws.append((pre, gi))
        vs_draws = {self.geoms[v]["draw"] for _p, v in draws}
        for did, lst in pre_by_draw.items():
            if did in vs_draws:
                continue                          # draw nay da co mesh Vs (Blender chi tao 1)
            for p in lst:
                draws.append((p, None))           # PreVs khong co Vs: van giu, khong bo mesh
        self.draws = draws
        self.active = 0 if not draws or self.draw_sel is None else max(0, min(self.draw_sel, len(draws) - 1))
        if draws:
            pre, vs = draws[self.active]
            g = self.geoms[vs if vs is not None else pre]
            self.draw_index, self.instances = g["draw"], g["inst"]
            self.geom = g
        else:
            self.geom = None

    @property
    def num_draws(self):
        return len(self.draws)

    @property
    def topology(self):
        return self.geom["topo"] if self.geom else TOPO_TRI

    @property
    def textures(self):
        """Texture cua rieng mesh nay (TXTR ma GEOM tham chieu)."""
        if not self.draws:
            return []
        pre, vs = self.draws[self.active]
        for gi in (vs, pre):
            if gi is not None and self.geoms[gi]["textures"]:
                return self.geoms[gi]["textures"]
        return []

    # --------------------------------------------------------------- helpers
    @property
    def _active_draw(self):
        if not self.draws:
            return (None, None)
        return self.draws[self.active]

    @property
    def input(self):
        """Buffer PreVs (toa do object-space) cua mesh dang chon."""
        pre, _vs = self._active_draw
        return self.buffers.get(pre) if pre is not None else None

    @property
    def outputs(self):
        """Buffer Vs (PostVs) cua mesh dang chon - dung canh tu SV_Position."""
        _pre, vs = self._active_draw
        return [self.buffers[vs]] if vs is not None else []

    @property
    def color_buffer(self):
        """Mau per-vertex lay tu buffer Vs (dung nhu Blender PostVS), khong co thi PreVs."""
        _pre, vs = self._active_draw
        if vs is not None and self.buffers[vs].attrs:
            return self.buffers[vs]
        return self.input

    @property
    def _any_buf(self):
        b = self.input
        if b is None and self.outputs:
            b = self.outputs[0]
        return b

    def vert_count(self):
        b = self._any_buf
        return b.vcount if b else 0

    def tri_count(self):
        b = self._any_buf
        return (b.index_count // 3) if b else 0

    def is_screen_quad(self):
        b = self._any_buf
        return b is not None and b.vcount <= 4

    def position_attr(self):
        b = self.input
        if not b:
            return None
        # Y het nrtools.createPreVsAuto: POSITION -> ATTRIBUTE 3 kenh -> attr[0]
        a = b.find(lambda a: a.name.upper() == "POSITION")
        if a is None:
            a = b.find(lambda a: a.name.upper() == "ATTRIBUTE" and a.comps == 3)
        if a is None and b.attrs:
            a = b.attrs[0]
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
    def local_geometry(self, uv_attr=None, with_color=False, color_label=None, color_index=None):
        """(positions Nx3, uvs Nx2|None, indices Mx3[, colors Nx3]) trong khong gian object."""
        b = self.input
        if b is None:
            # Mesh chi co Vs (khong co PreVs): dung lai toa do tu SV_Position
            # nhu Blender (createPos3FromPostVsAsList) roi tra ve.
            return self.view_geometry(60.0, 16 / 9, with_color=with_color,
                                      color_label=color_label, color_index=color_index) \
                if with_color else self.view_geometry(60.0, 16 / 9)
        pa = self.position_attr()
        if pa is None or b.indices is None:
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
            cols = mesh_vertex_colors(self, color_index=color_index, attr_label=color_label)
            if cols is not None:
                cols = cols[:len(pos)]
            return pos, uvs, idx, cols
        return pos, uvs, idx

    def view_geometry(self, fov_deg=60.0, aspect=16 / 9, uv_attr_label=None, clip_fn=None,
                      with_color=False, color_label=None, color_index=None):
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
        # Mau doc tu buffer PostVs (color_buffer) - noi chua mau that, map 1:1
        # theo chi so dinh voi output dang dung de dung canh.
        full_col = None
        if with_color:
            cb = self.color_buffer
            ci = color_index
            if ci == -1:                              # Auto: dò kenh mau that/mesh
                bi = best_color_index(self)
                ci = bi if bi >= 0 else None
            col_attr = _pick_color_attr(cb, ci, color_label)
            if col_attr is not None:
                fc = _read_color_blender(cb, col_attr)
                if fc is not None:
                    full_col = np.clip(fc[:, :3], 0.0, 1.0)
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
                if full_col is not None:
                    if len(full_col) == ob.vcount:            # PostVs = color_buffer, map 1:1
                        col = full_col
                    elif ob.vcount % len(full_col) == 0:      # instance lap lai
                        col = np.tile(full_col, (ob.vcount // len(full_col), 1))
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
    """Danh sach kenh vertex color (vc_0, vc_1, ...) Y HET importer Blender
    (nrtools.createVertexColorAttrsList):
      = [cac attr co semantic 'color']
      + [tat ca attr tu vi tri 1 tro di, bo qua vi tri 0 (position) va attr 'color' da them].
    KHONG loc theo kieu du lieu — Blender liet ke moi attr, nen so vc khop het voi Blender."""
    if buf is None or not buf.attrs:
        return []
    attrs = buf.attrs
    color_pos = [i for i, a in enumerate(attrs) if a.name.lower() == "color"]
    order = list(color_pos)
    for i in range(1, len(attrs)):
        if i in color_pos:
            continue
        order.append(i)
    return [attrs[i] for i in order]


def _read_color_blender(buf, attr):
    """Doc mau Y HET nrtools.unpackVertexColor:
      - Float  : doc thang cac thanh phan (KHONG chuan hoa).
      - Uint32 1 kenh : D3DCOLOR dong goi BGRA, chia 255.
      - Con lai (uint8/uint16/int...) : tra ve den (0) — dung nhu Blender.
    Tra ve (N,4) float."""
    if buf is None or attr is None or buf.raw is None or buf.vcount == 0:
        return None
    n, stride = buf.vcount, buf.stride
    arr = np.frombuffer(buf.raw, dtype=np.uint8).reshape(n, stride)
    c = np.zeros((n, 4), np.float32)
    if attr.fmt == 0:                                   # Float -> doc thang
        block = arr[:, attr.offset:attr.offset + 4 * attr.comps].copy()
        v = block.view(np.float32).reshape(n, attr.comps)
        k = min(attr.comps, 4)
        c[:, :k] = v[:, :k]
    elif attr.fmt == 1 and attr.comps == 1:             # Uint32 D3DCOLOR (BGRA)
        block = arr[:, attr.offset:attr.offset + 4].copy()
        col = block.view(np.uint32).reshape(n).astype(np.uint32)
        c[:, 0] = ((col >> 16) & 0xff) / 255.0           # R
        c[:, 1] = ((col >> 8) & 0xff) / 255.0            # G
        c[:, 2] = (col & 0xff) / 255.0                   # B
        c[:, 3] = ((col >> 24) & 0xff) / 255.0           # A
    # else: giu den — khop Blender (Uint8/Uint16 khong duoc unpack)
    return c


class _NRColorMixin:
    pass


def mesh_color_attrs(mesh):
    return _color_attrs(mesh.color_buffer)


def max_color_channels(meshes):
    return max((len(_color_attrs(m.color_buffer)) for m in meshes), default=0)


def best_color_index(mesh):
    """Tu dong chon kenh vc la MAU THAT giong Blender Base Color:
      - Chi nhan attr Float 3-4 thanh phan, gia tri trong [0,1] (bo normal uint8,
        bo UV float2 gia tri >1, bo cac kenh unpack ra den).
      - Uu tien kenh CO BIEN THIEN (mau that) hon kenh hang so (vd tint trang).
    Tra ve chi so vc (0..N-1); -1 neu mesh khong co kenh mau nao (chi co texture)."""
    b = mesh.color_buffer
    cands = _color_attrs(b)
    if not cands:
        return -1
    best_var, best_const = None, None
    for i, a in enumerate(cands):
        if a.fmt != 0 or a.comps not in (3, 4):      # chi mau float 3-4 kenh
            continue
        c = _read_color_blender(b, a)
        if c is None:
            continue
        rgb = c[:, :3]
        if rgb.max() > 1.001 or rgb.min() < -0.001:  # UV / du lieu khac, khong phai mau
            continue
        varies = len(np.unique(np.round(rgb, 3), axis=0)) > 1
        if varies and best_var is None:
            best_var = i
        elif not varies and best_const is None:
            best_const = i
    if best_var is not None:
        return best_var
    if best_const is not None:
        return best_const
    return -1


def mesh_vertex_colors(mesh, color_index=None, attr_label=None):
    """(N,3) float 0..1 hoac None. color_index = vc_N kieu Blender.
    color_index = -1 (Auto): tu chon kenh mau that giong Blender.
    Doc tu buffer PostVs (color_buffer) - noi chua mau that."""
    b = mesh.color_buffer
    cands = _color_attrs(b)
    if not cands:
        return None
    if color_index == -1:                            # Auto: dò kenh mau that
        bi = best_color_index(mesh)
        color_index = bi if bi >= 0 else None
    a = None
    if attr_label:
        a = next((x for x in cands if x.label == attr_label), None)
    if a is None and color_index is not None:
        a = cands[min(max(0, int(color_index)), len(cands) - 1)]
    if a is None:
        a = cands[0]
    c = _read_color_blender(b, a)
    if c is None:
        return None
    return np.clip(c[:, :3], 0.0, 1.0)


def _pick_color_attr(buf, color_index=None, color_label=None):
    cands = _color_attrs(buf)
    if not cands:
        return None
    if color_label:
        a = next((x for x in cands if x.label == color_label), None)
        if a is not None:
            return a
    if color_index is not None:
        return cands[min(max(0, int(color_index)), len(cands) - 1)]
    return cands[0]


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


def reference_correction(scene_pos, obj_pos, rotate_only=True):
    """
    Nan canh theo mesh chuan:
      rotate_only=True  -> CHI XOAY thang theo truc cua mesh goc (KHONG keo gian)
      rotate_only=False -> xoay + sua ca meo/ti le (khi FOV sai)
    scene_pos, obj_pos: (N,3) cung so dinh, cung thu tu.
    Tra ve (xform 3x3 de ap: pos @ xform.T, info).
    """
    S = np.asarray(scene_pos, np.float64)
    O = np.asarray(obj_pos, np.float64)
    n = min(len(S), len(O))
    S, O = S[:n], O[:n]
    Ps = S - S.mean(0)
    Po = O - O.mean(0)
    A, *_ = np.linalg.lstsq(Ps, Po, rcond=None)         # Ps @ A ~ Po
    U, sv, Vt = np.linalg.svd(A)
    R = U @ Vt                                          # phan xoay (co the gom lat guong)
    M = R if rotate_only else A
    resid = float(np.abs(Ps @ M - Po).max() / (np.abs(Po).max() + 1e-9))
    stretch = (sv / sv.max()).tolist() if sv.max() > 0 else [1.0, 1.0, 1.0]
    info = {"resid": resid, "stretch": stretch,
            "mirror": bool(np.linalg.det(R) < 0), "size": float(sv.mean())}
    return M.T.astype(np.float64), info


def face_normal_scene(pos, tri):
    """Phap tuyen chu dao cua 1 mat (trong khong gian canh) = tong phap tuyen
    tam giac co trong so dien tich. Tra ve vector don vi (3,) hoac None."""
    pos = np.asarray(pos, np.float64)
    tri = np.asarray(tri)
    if len(tri) == 0:
        return None
    v0, v1, v2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)          # do dai = 2*dien tich
    # Dinh huong nhat quan: lat cac phap tuyen nguoc ve cung ban cau truoc khi cong
    ref = n[np.argmax((n * n).sum(1))]
    s = np.sign((n * ref).sum(1))
    s[s == 0] = 1
    tot = (n * s[:, None]).sum(0)
    L = np.linalg.norm(tot)
    return (tot / L) if L > 1e-9 else None


def axis_align_rotation(normals):
    """Tu 2-3 phap tuyen mat skybox (khong gian canh) -> ma tran xoay 3x3 R
    sao cho cac mat VUONG GOC voi truc X/Y/Z. Chi xoay (truc giao, det=+1),
    KHONG keo gian. Ap dung: pos @ R.T. Chon hoan vi/dau truc gan huong hien
    tai nhat de khong lat lung tung."""
    import itertools
    N = [np.asarray(n, np.float64) for n in normals if n is not None]
    N = [n / np.linalg.norm(n) for n in N if np.linalg.norm(n) > 1e-9]
    if len(N) < 2:
        return None
    e1 = N[0]
    e2 = N[1] - np.dot(N[1], e1) * e1
    if np.linalg.norm(e2) < 1e-6:
        return None                         # 2 mat gan song song -> khong du thong tin
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    F = np.array([e1, e2, e3])              # he truc rieng cua skybox (trong canh), det=+1
    axes = np.eye(3)
    best, bestR = -1e9, None
    for perm in itertools.permutations(range(3)):
        for sgn in itertools.product((1, -1), repeat=3):
            T = np.array([axes[perm[0]] * sgn[0],
                          axes[perm[1]] * sgn[1],
                          axes[perm[2]] * sgn[2]])
            R = T.T @ F                      # R @ e_i = cot cua T -> mat thang truc
            if np.linalg.det(R) < 0:         # chi nhan xoay thuan (khong lat guong)
                continue
            tr = np.trace(R)                 # gan identity nhat = it phai xoay nhat
            if tr > best:
                best, bestR = tr, R
    return bestR


def object_faces_inverted(mesh):
    """Mesh object-space (PreVs) co bi nguoc mat khong?
    Canh dung lai tu SV_Position luon nguoc mat (dong nhat moi mesh, do quy uoc
    cull cua GPU). Khop affine object -> canh cho chinh mesh nay: det > 0 nghia la
    object cung chieu voi canh -> cung bi nguoc -> can Reverse. det < 0 -> dang dung."""
    try:
        if mesh.input is None:
            return True                         # mesh chi co Vs: toa do lay tu canh -> nguoc
        gl = mesh.local_geometry()
        gv = mesh.view_geometry(60.0, 16 / 9)
        if gl is None or gv is None:
            return False
        po, pv = np.asarray(gl[0], np.float64), np.asarray(gv[0], np.float64)
        k = min(len(po), len(pv))
        if k < 4:
            return False
        X = np.hstack([po[:k], np.ones((k, 1))])
        A, *_ = np.linalg.lstsq(X, pv[:k], rcond=None)
        d = np.linalg.det(A[:3, :3])
        return bool(d > 0) if abs(d) > 1e-12 else False
    except Exception:
        return False
