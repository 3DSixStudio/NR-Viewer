# -*- coding: utf-8 -*-
"""
Ghi file FBX nhi phan (FBX 7.4) thuan Python + numpy, khong can FBX SDK.
Mo duoc trong Maya, Blender, 3ds Max, Unity, Roblox Studio.

    write_fbx(path, meshes, embed_textures=False, copy_textures=True)
    meshes: list dict {name, pos (N,3), uv (N,2)|None, tri (M,3), texture (duong dan png)|None}
"""
import os
import shutil
import struct
import zlib

import numpy as np

_HEAD = b"Kaydara FBX Binary  \x00\x1a\x00"
_VERSION = 7400
_SENTINEL = b"\x00" * 13
# cac hang so nay phai khop nhau (FBX SDK kiem tra) - giong exporter cua Blender
_FILE_ID = b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1"
_TIME_ID = b"1970-01-01 10:00:00:000"
_FOOT_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"


# ----------------------------------------------------------------------------- node
class _Raw(bytes):
    pass


class _Long(int):
    """So nguyen 64-bit (ID doi tuong FBX luon la kieu L)."""


class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, *props):
        self.name = name
        self.props = list(props)
        self.children = []

    def add(self, name, *props):
        n = Node(name, *props)
        self.children.append(n)
        return n

    # ---- encode
    @staticmethod
    def _prop(v):
        if isinstance(v, _Raw):
            return b"R" + struct.pack("<I", len(v)) + bytes(v)
        if isinstance(v, _Long):
            return b"L" + struct.pack("<q", int(v))
        if isinstance(v, bool):
            return b"C" + struct.pack("<B", 1 if v else 0)
        if isinstance(v, int):
            if -2**31 <= v < 2**31:
                return b"I" + struct.pack("<i", v)
            return b"L" + struct.pack("<q", v)
        if isinstance(v, float):
            return b"D" + struct.pack("<d", v)
        if isinstance(v, str):
            b = v.encode("utf-8")
            return b"S" + struct.pack("<I", len(b)) + b
        if isinstance(v, bytes):
            return b"S" + struct.pack("<I", len(v)) + v
        if isinstance(v, np.ndarray):
            code = {np.dtype(np.float64): b"d", np.dtype(np.float32): b"f",
                    np.dtype(np.int32): b"i", np.dtype(np.int64): b"l"}[v.dtype]
            data = np.ascontiguousarray(v).tobytes()
            if len(data) > 256:
                comp = zlib.compress(data, 6)
                return code + struct.pack("<III", v.size, 1, len(comp)) + comp
            return code + struct.pack("<III", v.size, 0, len(data)) + data
        raise TypeError(type(v))

    def encode(self, offset):
        props = b"".join(self._prop(p) for p in self.props)
        name = self.name.encode()
        head_len = 13 + len(name)
        body_start = offset + head_len + len(props)
        body = b""
        if self.children:
            parts = []
            pos = body_start
            for c in self.children:
                e = c.encode(pos)
                parts.append(e)
                pos += len(e)
            body = b"".join(parts) + _SENTINEL
        elif not self.props:
            body = _SENTINEL
        end = body_start + len(body)
        return struct.pack("<IIIB", end, len(self.props), len(props), len(name)) + name + props + body


def _P(parent, name, typ, label, flags, *vals):
    return parent.add("P", name, typ, label, flags, *vals)


# ----------------------------------------------------------------------------- helpers
def _flat_normals(pos, tri):
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    n = np.cross(b - a, c - a)
    L = np.linalg.norm(n, axis=1, keepdims=True)
    L[L == 0] = 1
    n = n / L
    return np.repeat(n, 3, axis=0)            # ByPolygonVertex


class _Ids:
    def __init__(self):
        self.n = 1000000

    def __call__(self):
        self.n += 1
        return _Long(self.n)


def _relpath(p, start):
    """relpath an toan tren Windows: khac o dia (C: / D:) thi tra ve duong dan tuyet doi."""
    try:
        return os.path.relpath(p, start).replace("\\", "/")
    except ValueError:
        return os.path.abspath(p).replace("\\", "/")


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name) or "obj"


# ----------------------------------------------------------------------------- main
def write_fbx(path, meshes, embed_textures=False, copy_textures=True, creator="NR Viewer"):
    """Moi mesh co the co m["flip"]=True: dao chieu mat (= Maya Reverse).
    Normal luon ghi phang theo tung mat (= Maya Normals > Set to Face)."""
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)
    tex_dir_name = os.path.splitext(os.path.basename(path))[0] + "_textures"
    tex_dir = os.path.join(base_dir, tex_dir_name)
    nid = _Ids()

    # ---- thu thap material: texture (theo duong dan anh) va mau (theo vertex color)
    tex_files = {}      # tex src -> (abs dst, rel)
    if copy_textures:
        for m in meshes:
            t = m.get("texture")
            if t and os.path.isfile(t) and t not in tex_files:
                os.makedirs(tex_dir, exist_ok=True)
                dst = os.path.join(tex_dir, os.path.basename(t))
                if os.path.abspath(dst) != os.path.abspath(t):
                    shutil.copyfile(t, dst)
                tex_files[t] = (dst, tex_dir_name + "/" + os.path.basename(t))

    tex_mats = {}       # tex_path|None -> dict(mat, tex, vid, name)
    col_mats = {}       # (r,g,b 0-255) -> mat_id
    objects = Node("Objects")
    conns = Node("Connections")
    n_geom = n_model = 0

    def _use_color(m):
        cols = m.get("colors")
        fm = m.get("face_mat")
        return cols is not None and fm is not None and len(cols) > 0

    used_names = {}
    for m in meshes:
        pos = np.asarray(m["pos"], np.float64)
        tri = np.asarray(m["tri"], np.int64)
        if len(tri) == 0 or len(pos) == 0:
            continue
        if m.get("flip"):
            tri = tri[:, ::-1].copy()      # Reverse: dao thu tu dinh -> mat quay ra ngoai
        # Normal ghi ben duoi = _flat_normals(tri da dao) -> y het "Set to Face"
        name = _safe(m["name"])
        # ten phai duy nhat (tranh Maya gop node trung ten); trung thi them hau to
        # kieu Blender (ten, ten.001...) nhung Maya cam dau cham -> ten_001, ten_002
        base = name
        if base in used_names:
            used_names[base] += 1
            name = f"{base}_{used_names[base]:03d}"
            while name in used_names:                 # dam bao khong dung voi ten co san
                used_names[base] += 1
                name = f"{base}_{used_names[base]:03d}"
        else:
            used_names[base] = 0
        used_names[name] = 0                          # danh dau ten cuoi cung da dung
        gid, mid = nid(), nid()

        # material cua mesh nay (theo thu tu local)
        color_mode = _use_color(m)
        if color_mode:
            local_ids = []
            for c in m["colors"]:
                key = tuple(int(max(0, min(255, round(float(x) * 255)))) for x in c[:3])
                if key not in col_mats:
                    col_mats[key] = nid()
                local_ids.append(col_mats[key])
            face_mat = np.asarray(m["face_mat"], np.int32)
        else:
            t = m.get("texture")
            if t and not os.path.isfile(t):
                t = None
            if t not in tex_mats:
                nm = "M_" + (_safe(os.path.splitext(os.path.basename(t))[0]) if t else "NoTex")
                tex_mats[t] = dict(mat=nid(), tex=nid() if t else None,
                                   vid=nid() if t else None, name=nm, path=t)
            local_ids = [tex_mats[t]["mat"]]

        g = objects.add("Geometry", gid, f"{name}\x00\x01Geometry", "Mesh")
        g.add("Properties70")
        g.add("GeometryVersion", 124)
        g.add("Vertices", pos.ravel().astype(np.float64))
        pvi = tri.astype(np.int32).copy()
        pvi[:, 2] = -pvi[:, 2] - 1
        g.add("PolygonVertexIndex", pvi.ravel())

        ln = g.add("LayerElementNormal", 0)
        ln.add("Version", 101)
        ln.add("Name", "")
        ln.add("MappingInformationType", "ByPolygonVertex")
        ln.add("ReferenceInformationType", "Direct")
        ln.add("Normals", _flat_normals(pos, tri).ravel().astype(np.float64))

        uv = m.get("uv")
        if uv is not None:
            uv = np.asarray(uv, np.float64).copy()
            uv[:, 1] = 1.0 - uv[:, 1]
            lu = g.add("LayerElementUV", 0)
            lu.add("Version", 101)
            lu.add("Name", "UVMap")
            lu.add("MappingInformationType", "ByPolygonVertex")
            lu.add("ReferenceInformationType", "IndexToDirect")
            lu.add("UV", uv.ravel())
            lu.add("UVIndex", tri.astype(np.int32).ravel())

        lm = g.add("LayerElementMaterial", 0)
        lm.add("Version", 101)
        lm.add("Name", "")
        if color_mode and len(local_ids) > 1:
            lm.add("MappingInformationType", "ByPolygon")
            lm.add("ReferenceInformationType", "IndexToDirect")
            lm.add("Materials", face_mat)
        else:
            lm.add("MappingInformationType", "AllSame")
            lm.add("ReferenceInformationType", "IndexToDirect")
            lm.add("Materials", np.zeros(1, np.int32))

        lay = g.add("Layer", 0)
        lay.add("Version", 100)
        for typ in (["LayerElementNormal"] + (["LayerElementUV"] if uv is not None else []) + ["LayerElementMaterial"]):
            le = lay.add("LayerElement")
            le.add("Type", typ)
            le.add("TypedIndex", 0)

        mo = objects.add("Model", mid, f"{name}\x00\x01Model", "Mesh")
        mo.add("Version", 232)
        p70 = mo.add("Properties70")
        _P(p70, "DefaultAttributeIndex", "int", "Integer", "", 0)
        _P(p70, "Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0)
        _P(p70, "Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0)
        _P(p70, "Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)
        mo.add("MultiLayer", 0)
        mo.add("MultiTake", 0)
        mo.add("Shading", True)
        mo.add("Culling", "CullingOff")

        conns.add("C", "OO", gid, mid)
        conns.add("C", "OO", mid, _Long(0))
        for mat_id in local_ids:                    # thu tu ket noi = chi so material local
            conns.add("C", "OO", mat_id, mid)
        n_geom += 1
        n_model += 1

    # ---- phat material texture
    n_tex = 0
    for t, info in tex_mats.items():
        mat_id, mname = info["mat"], info["name"]
        ma = objects.add("Material", mat_id, f"{mname}\x00\x01Material", "")
        ma.add("Version", 102)
        ma.add("ShadingModel", "phong")
        ma.add("MultiLayer", 0)
        p70 = ma.add("Properties70")
        _P(p70, "DiffuseColor", "Color", "", "A", 1.0, 1.0, 1.0)
        _P(p70, "DiffuseFactor", "Number", "", "A", 1.0)
        _P(p70, "SpecularFactor", "Number", "", "A", 0.0)
        _P(p70, "Shininess", "double", "Number", "", 2.0)
        _P(p70, "ShininessExponent", "Number", "", "A", 2.0)
        _P(p70, "Diffuse", "Vector3D", "Vector", "", 1.0, 1.0, 1.0)
        if not t:
            continue
        if t in tex_files:
            absf, rel = tex_files[t]
        else:
            absf, rel = t, _relpath(t, base_dir)
        tname = _safe(os.path.splitext(os.path.basename(t))[0])
        tex_id, vid_id = info["tex"], info["vid"]

        vi = objects.add("Video", vid_id, f"{tname}\x00\x01Video", "Clip")
        vi.add("Type", "Clip")
        vp = vi.add("Properties70")
        _P(vp, "Path", "KString", "XRefUrl", "", absf)
        vi.add("UseMipMap", 0)
        vi.add("Filename", absf)
        vi.add("RelativeFilename", rel)
        if embed_textures:
            with open(t, "rb") as f:
                vi.add("Content", _Raw(f.read()))

        tx = objects.add("Texture", tex_id, f"{tname}\x00\x01Texture", "")
        tx.add("Type", "TextureVideoClip")
        tx.add("Version", 202)
        tx.add("TextureName", f"{tname}\x00\x01Texture")
        tp = tx.add("Properties70")
        _P(tp, "UVSet", "KString", "", "", "UVMap")
        _P(tp, "UseMaterial", "bool", "", "", 1)
        tx.add("Media", f"{tname}\x00\x01Video")
        tx.add("FileName", absf)
        tx.add("RelativeFilename", rel)
        tx.add("ModelUVTranslation", 0.0, 0.0)
        tx.add("ModelUVScaling", 1.0, 1.0)
        tx.add("Texture_Alpha_Source", "None")
        tx.add("Cropping", 0, 0, 0, 0)
        conns.add("C", "OP", tex_id, mat_id, "DiffuseColor")
        conns.add("C", "OO", vid_id, tex_id)
        n_tex += 1

    # ---- phat material mau (tu vertex color)
    for key, mat_id in col_mats.items():
        r, gg, bb = (key[0] / 255.0, key[1] / 255.0, key[2] / 255.0)
        mname = "C_%02X%02X%02X" % key
        ma = objects.add("Material", mat_id, f"{mname}\x00\x01Material", "")
        ma.add("Version", 102)
        ma.add("ShadingModel", "phong")
        ma.add("MultiLayer", 0)
        p70 = ma.add("Properties70")
        _P(p70, "DiffuseColor", "Color", "", "A", r, gg, bb)
        _P(p70, "DiffuseFactor", "Number", "", "A", 1.0)
        _P(p70, "SpecularFactor", "Number", "", "A", 0.0)
        _P(p70, "Shininess", "double", "Number", "", 2.0)
        _P(p70, "Diffuse", "Vector3D", "Vector", "", r, gg, bb)

    n_mats = len(tex_mats) + len(col_mats)

    # ---- header / settings
    top = []
    hx = Node("FBXHeaderExtension")
    hx.add("FBXHeaderVersion", 1003)
    hx.add("FBXVersion", _VERSION)
    hx.add("EncryptionType", 0)
    ts = hx.add("CreationTimeStamp")
    for k, v in (("Version", 1000), ("Year", 1970), ("Month", 1), ("Day", 1), ("Hour", 10),
                 ("Minute", 0), ("Second", 0), ("Millisecond", 0)):
        ts.add(k, v)
    hx.add("Creator", creator)
    top.append(hx)
    top.append(Node("FileId", _Raw(_FILE_ID)))
    top.append(Node("CreationTime", _TIME_ID.decode()))
    top.append(Node("Creator", creator))

    gs = Node("GlobalSettings")
    gs.add("Version", 1000)
    p70 = gs.add("Properties70")
    for k, v in (("UpAxis", 1), ("UpAxisSign", 1), ("FrontAxis", 2), ("FrontAxisSign", 1),
                 ("CoordAxis", 0), ("CoordAxisSign", 1), ("OriginalUpAxis", 1), ("OriginalUpAxisSign", 1)):
        _P(p70, k, "int", "Integer", "", v)
    _P(p70, "UnitScaleFactor", "double", "Number", "", 1.0)
    _P(p70, "OriginalUnitScaleFactor", "double", "Number", "", 1.0)
    top.append(gs)

    docs = Node("Documents")
    docs.add("Count", 1)
    doc = docs.add("Document", nid(), "", "Scene")
    doc.add("Properties70")
    doc.add("RootNode", _Long(0))
    top.append(docs)
    top.append(Node("References"))

    defs = Node("Definitions")
    defs.add("Version", 100)
    counts = [("GlobalSettings", 1), ("Geometry", n_geom), ("Model", n_model),
              ("Material", n_mats), ("Texture", n_tex), ("Video", n_tex)]
    defs.add("Count", sum(c for _, c in counts))
    for typ, c in counts:
        if c:
            ot = defs.add("ObjectType", typ)
            ot.add("Count", c)
    top.append(defs)
    top.append(objects)
    top.append(conns)

    # ---- ghi
    out = bytearray(_HEAD + struct.pack("<I", _VERSION))
    for n in top:
        out += n.encode(len(out))
    out += _SENTINEL
    out += _FOOT_ID
    out += b"\x00" * 4
    pad = ((len(out) + 15) & ~15) - len(out)
    out += b"\x00" * (pad or 16)
    out += struct.pack("<I", _VERSION)
    out += b"\x00" * 120
    out += b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
    with open(path, "wb") as f:
        f.write(out)
    return {"models": n_model, "materials": n_mats, "textures": n_tex,
            "tex_dir": tex_dir if tex_files else None}
