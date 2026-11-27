# -*- coding: utf-8 -*-
"""Viewport OpenGL 3.3 core cho NR Viewer (PySide6 + PyOpenGL)."""
import ctypes
import math

import numpy as np
import OpenGL
# Tat kiem tra loi cua PyOpenGL: mot so driver dat GL error vo hai (vd glPolygonMode,
# line width...) khien PyOpenGL nem exception lam paintGL dung giua chung -> viewport trang.
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = False
from OpenGL import GL
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QImage, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLFramebufferObject

# --------------------------------------------------------------------------- shaders
MESH_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
layout(location=2) in vec3 aCol;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vViewPos;
out vec2 vUV;
out vec3 vCol;
void main(){
    vec4 vp = uView * uModel * vec4(aPos,1.0);
    vViewPos = vp.xyz;
    vUV = aUV;
    vCol = aCol;
    gl_Position = uProj * vp;
}
"""

MESH_FS = """
#version 330 core
in vec3 vViewPos;
in vec2 vUV;
in vec3 vCol;
uniform int  uMode;        // 0 texture, 1 clay, 2 normal, 3 uv checker, 4 vertex color
uniform int  uHasTex;
uniform sampler2D uTex;
uniform float uAlphaCut;
uniform vec3 uTint;
uniform int  uWire;        // 1 = pass ve wireframe
uniform int  uSel;
out vec4 frag;
void main(){
    if(uWire==1){ frag = vec4(0.05,0.85,1.0,1.0); return; }
    if(uWire==2){ frag = vec4(uTint,1.0); return; }   // pick ID
    vec3 n = normalize(cross(dFdx(vViewPos), dFdy(vViewPos)));
    float lam = abs(dot(n, normalize(-vViewPos)));
    float shade = 0.35 + 0.65*lam;
    vec3 base = vec3(0.72);
    float a = 1.0;
    if(uMode==0 && uHasTex==1){
        vec4 t = texture(uTex, vUV);
        if(t.a < uAlphaCut) discard;
        base = t.rgb; a = 1.0;
        shade = 0.55 + 0.45*lam;
    } else if(uMode==2){
        frag = vec4(n*0.5+0.5,1.0); return;
    } else if(uMode==3){
        vec2 c = floor(fract(vUV)*8.0);
        float k = mod(c.x+c.y,2.0);
        base = mix(vec3(0.9,0.45,0.15), vec3(0.95), k);
    } else if(uMode==4){
        base = vCol;
        shade = 0.55 + 0.45*lam;
    }
    vec3 col = base*shade;
    if(uSel==1) col = mix(col, vec3(1.0,0.55,0.1), 0.5);
    frag = vec4(col, a);
}
"""

LINE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aCol;
uniform mat4 uView; uniform mat4 uProj;
out vec3 vCol;
void main(){ vCol=aCol; gl_Position = uProj*uView*vec4(aPos,1.0); }
"""
LINE_FS = """
#version 330 core
in vec3 vCol; out vec4 frag;
void main(){ frag=vec4(vCol,1.0); }
"""

BG_VS = """
#version 330 core
layout(location=0) in vec2 p;
out float vy;
void main(){ vy = p.y*0.5+0.5; gl_Position = vec4(p,0.0,1.0); }
"""
BG_FS = """
#version 330 core
in float vy; out vec4 frag;
uniform vec3 uTop; uniform vec3 uBot;
void main(){ frag = vec4(mix(uBot, uTop, vy), 1.0); }
"""


def _compile(vs, fs):
    def sh(src, kind):
        s = GL.glCreateShader(kind)
        GL.glShaderSource(s, src)
        GL.glCompileShader(s)
        if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
        return s
    p = GL.glCreateProgram()
    a, b = sh(vs, GL.GL_VERTEX_SHADER), sh(fs, GL.GL_FRAGMENT_SHADER)
    GL.glAttachShader(p, a)
    GL.glAttachShader(p, b)
    GL.glLinkProgram(p)
    if not GL.glGetProgramiv(p, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(p).decode())
    GL.glDeleteShader(a)
    GL.glDeleteShader(b)
    return p


# --------------------------------------------------------------------------- math
def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.asarray(v, np.float64), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(f, (0, 0, 1))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m.astype(np.float32)


def translate(t):
    m = np.identity(4, np.float32)
    m[:3, 3] = t
    return m


def _euler_xyz(r_deg):
    rx, ry, rz = np.radians(r_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def trs_matrix(center, t, r_deg, s):
    """Ma tran 4x4: Scale + Rotate + Translate quanh diem center (world)."""
    c = np.asarray(center, np.float64)
    M = np.identity(4)
    M[:3, :3] = _euler_xyz(r_deg) @ np.diag(np.asarray(s, np.float64))
    M[:3, 3] = c + np.asarray(t, np.float64) - M[:3, :3] @ c
    return M.astype(np.float32)


def apply_trs(pos, t, r_deg, s):
    """Bien doi mang dinh Nx3 quanh tam cua chinh no."""
    pos = np.asarray(pos, np.float64)
    c = (pos.min(0) + pos.max(0)) / 2
    M = trs_matrix(c, t, r_deg, s).astype(np.float64)
    return (pos @ M[:3, :3].T + M[:3, 3]).astype(np.float32)


# --------------------------------------------------------------------------- GPU mesh
class GpuMesh:
    def __init__(self, key, pos, uv, tri, tex_path, offset=(0, 0, 0), vcol=None):
        self.key = key
        self.tex_path = tex_path
        self.count = int(tri.size)
        self.bmin = pos.min(0) + offset
        self.bmax = pos.max(0) + offset
        self.offset = np.asarray(offset, np.float64)
        self.base = translate(offset)
        self.model = self.base
        self.trs = None                 # (t, r, s) nguoi dung dat
        self.tint = (1.0, 1.0, 1.0)
        self.visible = True
        if uv is None:
            uv = np.zeros((len(pos), 2), np.float32)
        if vcol is None:
            vcol = np.ones((len(pos), 3), np.float32)
        else:
            vcol = np.asarray(vcol, np.float32)[:, :3]
            if len(vcol) < len(pos):
                vcol = np.vstack([vcol, np.ones((len(pos) - len(vcol), 3), np.float32)])
        data = np.hstack([pos.astype(np.float32), uv.astype(np.float32), vcol[:len(pos)]]).ravel()
        self.vao = GL.glGenVertexArrays(1)
        self.vbo, self.ebo = GL.glGenBuffers(2)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        idx = np.ascontiguousarray(tri, dtype=np.uint32)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 32, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, 32, ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, False, 32, ctypes.c_void_p(20))
        GL.glBindVertexArray(0)

    def set_user_transform(self, t, r_deg, s):
        if t == (0, 0, 0) and r_deg == (0, 0, 0) and s == (1, 1, 1):
            self.trs = None
            self.model = self.base
            return
        self.trs = (t, r_deg, s)
        c = (self.bmin + self.bmax) / 2
        self.model = (trs_matrix(c, t, r_deg, s) @ self.base).astype(np.float32)

    def draw(self):
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self.count, GL.GL_UNSIGNED_INT, None)

    def release(self):
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(2, [self.vbo, self.ebo])


# --------------------------------------------------------------------------- widget
class GLViewport(QOpenGLWidget):
    statsChanged = Signal(str)
    meshPicked = Signal(str, bool)   # ten mesh ("" = trong), them vao selection?
    frameRequested = Signal()        # phim F: zoom vao vat dang chon
    backgroundChanged = Signal(str)  # Ctrl+B doi nen
    transformChanged = Signal(str, object, object, object)  # key, t, r, s (do keo gizmo)
    transformBegin = Signal(str, object, object, object)    # truoc khi keo (de undo)
    transformEnd = Signal(str, object, object, object)      # sau khi tha (de undo)
    gizmoModeChanged = Signal(str)   # 'move' / 'rotate' / 'scale'

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        self.setFormat(fmt)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)

        self.meshes = []            # GpuMesh
        self._pending = []          # (key, pos, uv, tri, tex, offset) cho upload
        self._tex_cache = {}
        self._clear_req = False

        self.mode = 0
        self.wire = False
        self.cull = False
        self.alpha_cut = 0.1
        self.show_grid = True
        self.fov = 60.0
        self.bg_list = [
            ((0.22, 0.24, 0.29), (0.09, 0.09, 0.11), "Gradient"),
            ((0.13, 0.13, 0.15), (0.13, 0.13, 0.15), "Xám đậm"),
            ((0.36, 0.36, 0.39), (0.36, 0.36, 0.39), "Xám"),
            ((0.70, 0.70, 0.72), (0.70, 0.70, 0.72), "Sáng"),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "Đen"),
        ]
        self.bg_index = 0

        self.yaw, self.pitch, self.dist = 35.0, -25.0, 10.0
        self.target = np.zeros(3, np.float32)
        self._last = QPoint()
        self._ready = False

        # --- gizmo (cong cu di chuyen / xoay / scale tren vat dang chon) ---
        self.gizmo_mode = "move"       # 'move' | 'rotate' | 'scale' | None
        self.gizmo_key = None
        self.giz_vao = None
        self._giz_axis = None          # truc dang keo (0,1,2) hoac 3 = tam (uniform)
        self._giz_snap = None          # (t0, r0, s0, center_screen, [tip_screen...])

    # ------------------------------------------------------------------ API
    def clear(self):
        self._pending.clear()
        self._clear_req = True
        self.update()

    def add_mesh(self, key, pos, uv, tri, tex_path, offset=(0, 0, 0), vcol=None):
        if tri is None or len(tri) == 0:
            return
        self._pending.append((key, pos, uv, tri, tex_path, np.asarray(offset, np.float32), vcol))
        self.update()

    def set_transform(self, key, t, r_deg, s):
        for m in self.meshes:
            if m.key == key:
                m.set_user_transform(t, r_deg, s)
        self.update()

    def mesh_center(self, key):
        for m in self.meshes:
            if m.key == key:
                return ((m.bmin + m.bmax) / 2).astype(np.float32)
        return None

    def set_tint(self, keys):
        keys = set(keys)
        for m in self.meshes:
            m.tint = (1.0, 0.75, 0.45) if m.key in keys else (1.0, 1.0, 1.0)
        self.update()

    def set_hidden(self, keys):
        keys = set(keys)
        for m in self.meshes:
            m.visible = m.key not in keys
        self._emit_stats()
        self.update()

    def loaded_keys(self):
        return [m.key for m in self.meshes]

    def bounds(self, only=None):
        ms = [m for m in self.meshes if m.visible and (only is None or m.key in only)]
        if not ms:
            return None
        mn = np.min([m.bmin for m in ms], axis=0)
        mx = np.max([m.bmax for m in ms], axis=0)
        return mn, mx

    def frame_all(self, only=None):
        b = self.bounds(only) or self.bounds()
        if b is None:
            return
        mn, mx = b
        self.target = ((mn + mx) / 2).astype(np.float32)
        r = float(np.linalg.norm(mx - mn)) / 2 or 1.0
        self.dist = r / math.tan(math.radians(self.fov) / 2) * 1.1
        self.update()

    def set_camera(self, eye, fwd, fov=None, dist=None):
        """Dat camera tai eye, nhin theo huong fwd (world)."""
        fwd = np.asarray(fwd, np.float64)
        fwd /= np.linalg.norm(fwd) + 1e-12
        if fov:
            self.fov = float(fov)
        if dist is None:
            b = self.bounds()
            dist = 10.0
            if b is not None:
                c = (b[0] + b[1]) / 2
                dist = max(1.0, float(np.dot(c - eye, fwd)))
        d = -fwd
        self.yaw = math.degrees(math.atan2(d[0], d[2]))
        self.pitch = -math.degrees(math.asin(max(-1.0, min(1.0, d[1]))))
        self.dist = dist
        self.target = (np.asarray(eye) + fwd * dist).astype(np.float32)
        self.update()

    def camera_origin(self, fov=None):
        """Dat camera vao vi tri camera goc cua game (0,0,0) nhin -Z."""
        if fov:
            self.fov = fov
        b = self.bounds()
        d = 10.0
        if b is not None:
            d = max(1.0, float(-(b[0][2] + b[1][2]) / 2))
        self.yaw, self.pitch, self.dist = 0.0, 0.0, d
        self.target = np.array([0, 0, -d], np.float32)
        self.update()

    # ----------------------------------------------------------- GL events
    def initializeGL(self):
        self.prog = _compile(MESH_VS, MESH_FS)
        self.lprog = _compile(LINE_VS, LINE_FS)
        self.bgprog = _compile(BG_VS, BG_FS)
        quad = np.array([-1, -1, 3, -1, -1, 3], np.float32)   # tam giac phu man hinh
        self.bg_vao = GL.glGenVertexArrays(1)
        self.bg_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.bg_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.bg_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 8, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)
        self.grid_vao, self.grid_n = None, 0
        self._build_grid(10.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self._white = self._make_tex(np.full((1, 1, 4), 255, np.uint8))
        self._ready = True

    def _make_tex(self, rgba):
        h, w = rgba.shape[:2]
        t = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        return t

    def _texture(self, path):
        if not path:
            return None
        if path in self._tex_cache:
            return self._tex_cache[path]
        img = QImage(path)
        tid = None
        if not img.isNull():
            img = img.convertToFormat(QImage.Format_RGBA8888)
            w, h = img.width(), img.height()
            arr = np.frombuffer(img.constBits(), np.uint8, img.bytesPerLine() * h)
            arr = arr.reshape(h, img.bytesPerLine())[:, :w * 4].reshape(h, w, 4).copy()
            tid = self._make_tex(arr)
        self._tex_cache[path] = tid
        return tid

    def _build_grid(self, size):
        n = 20
        step = size / n * 2
        lines, cols = [], []
        for i in range(-n, n + 1):
            c = (0.32, 0.32, 0.36) if i else (0.5, 0.5, 0.55)
            lines += [(-size, 0, i * step / 2), (size, 0, i * step / 2), (i * step / 2, 0, -size), (i * step / 2, 0, size)]
            cols += [c] * 4
        lines += [(0, 0, 0), (size * 0.5, 0, 0), (0, 0, 0), (0, size * 0.5, 0), (0, 0, 0), (0, 0, size * 0.5)]
        cols += [(0.9, 0.25, 0.25)] * 2 + [(0.3, 0.9, 0.3)] * 2 + [(0.3, 0.5, 1.0)] * 2
        data = np.hstack([np.array(lines, np.float32), np.array(cols, np.float32)]).ravel()
        if self.grid_vao is None:
            self.grid_vao = GL.glGenVertexArrays(1)
            self.grid_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.grid_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 24, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 24, ctypes.c_void_p(12))
        GL.glBindVertexArray(0)
        self.grid_n = len(lines)
        self._grid_size = size

    def _flush(self):
        if self._clear_req:
            for m in self.meshes:
                m.release()
            self.meshes.clear()
            self._clear_req = False
        if self._pending:
            for key, pos, uv, tri, tex, off, vcol in self._pending:
                self.meshes.append(GpuMesh(key, pos, uv, tri, tex, off, vcol))
            self._pending.clear()
            b = self.bounds()
            if b is not None:
                s = float(np.max(np.abs(np.concatenate(b)))) or 1.0
                s = 10 ** math.ceil(math.log10(s))
                if s != self._grid_size:
                    self._build_grid(s)
            self._emit_stats()

    def _emit_stats(self):
        vis = [m for m in self.meshes if m.visible]
        tris = sum(m.count for m in vis) // 3
        hid = len(self.meshes) - len(vis)
        self.statsChanged.emit(f"{len(vis)} mesh | {tris:,} tam giác" + (f" | đang ẩn {hid}" if hid else ""))

    def view_matrix(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        d = np.array([math.cos(p) * math.sin(y), math.sin(-p), math.cos(p) * math.cos(y)])
        eye = self.target + d * self.dist
        return look_at(eye, self.target)

    def _matrices(self, w, h):
        b = self.bounds()
        rad = float(np.linalg.norm(b[1] - b[0])) if b is not None else 10.0
        far = (self.dist + rad + float(np.linalg.norm(self.target))) * 2 + 10
        # near lon hon -> do chinh xac depth tot hon (tranh nhieu/z-fighting o xa)
        near = max(far / 20000.0, self.dist * 0.01, 0.005)
        return self.view_matrix(), perspective(self.fov, w / h, near, far)

    def _draw_meshes(self, view, proj, pick=False):
        p = self.prog
        GL.glUseProgram(p)
        loc = lambda n: GL.glGetUniformLocation(p, n)
        GL.glUniformMatrix4fv(loc("uView"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(loc("uProj"), 1, GL.GL_TRUE, proj)
        GL.glUniform1i(loc("uMode"), self.mode)
        GL.glUniform1f(loc("uAlphaCut"), self.alpha_cut)
        GL.glUniform1i(loc("uTex"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        for i, m in enumerate(self.meshes):
            if not m.visible:
                continue
            GL.glUniformMatrix4fv(loc("uModel"), 1, GL.GL_TRUE, m.model)
            tid = self._texture(m.tex_path)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tid or self._white)
            GL.glUniform1i(loc("uHasTex"), 1 if tid else 0)
            if pick:
                k = i + 1
                GL.glUniform3f(loc("uTint"), (k & 255) / 255.0, ((k >> 8) & 255) / 255.0, ((k >> 16) & 255) / 255.0)
                GL.glUniform1i(loc("uSel"), 0)
                GL.glUniform1i(loc("uWire"), 2)
                m.draw()
                continue
            GL.glUniform1i(loc("uSel"), 1 if m.tint != (1.0, 1.0, 1.0) else 0)
            GL.glUniform1i(loc("uWire"), 0)
            if self.wire:
                GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
                GL.glPolygonOffset(1.0, 1.0)
            m.draw()
            if self.wire:
                GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
                GL.glUniform1i(loc("uWire"), 1)
                m.draw()
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glBindVertexArray(0)

    def paintGL(self):
        try:
            self._paint()
        except Exception as e:  # noqa - khong de loi ve lam trang viewport
            import traceback
            traceback.print_exc()

    def _paint(self):
        self._flush()
        dpr = self.devicePixelRatioF()
        w, h = max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr))
        GL.glViewport(0, 0, w, h)
        top, bot, _ = self.bg_list[self.bg_index]
        GL.glClearColor(bot[0], bot[1], bot[2], 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # ve nen (gradient hoac mau dac) phu toan man hinh
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self.bgprog)
        GL.glUniform3f(GL.glGetUniformLocation(self.bgprog, "uTop"), *top)
        GL.glUniform3f(GL.glGetUniformLocation(self.bgprog, "uBot"), *bot)
        GL.glBindVertexArray(self.bg_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        view, proj = self._matrices(w, h)

        if self.show_grid:
            GL.glUseProgram(self.lprog)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.lprog, "uView"), 1, GL.GL_TRUE, view)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.lprog, "uProj"), 1, GL.GL_TRUE, proj)
            GL.glBindVertexArray(self.grid_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self.grid_n)

        if self.cull:
            GL.glEnable(GL.GL_CULL_FACE)
        else:
            GL.glDisable(GL.GL_CULL_FACE)
        self._draw_meshes(view, proj)
        self._draw_gizmo(view, proj)

    def pick(self, x, y):
        """Tra ve ten mesh tai diem (x, y) toa do widget, hoac ''."""
        if not self.meshes:
            return ""
        self.makeCurrent()
        self._flush()
        dpr = self.devicePixelRatioF()
        w, h = max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr))
        fbo = QOpenGLFramebufferObject(w, h, QOpenGLFramebufferObject.Depth)
        fbo.bind()
        GL.glViewport(0, 0, w, h)
        GL.glDisable(GL.GL_MULTISAMPLE)
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        if self.cull:
            GL.glEnable(GL.GL_CULL_FACE)
        else:
            GL.glDisable(GL.GL_CULL_FACE)
        view, proj = self._matrices(w, h)
        self._draw_meshes(view, proj, pick=True)
        px = GL.glReadPixels(int(x * dpr), h - 1 - int(y * dpr), 1, 1, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        fbo.release()
        GL.glEnable(GL.GL_MULTISAMPLE)
        self.doneCurrent()
        b = bytes(px) if not isinstance(px, np.ndarray) else px.tobytes()
        k = b[0] | (b[1] << 8) | (b[2] << 16)
        self.update()
        return self.meshes[k - 1].key if 0 < k <= len(self.meshes) else ""

    # ------------------------------------------------------------ input
    def mousePressEvent(self, e):
        self._last = e.position().toPoint()
        self._press = self._last
        self._giz_dragging = False
        if e.button() == Qt.LeftButton and not (e.modifiers() & Qt.ShiftModifier):
            p = e.position().toPoint()
            if self._gizmo_press(np.array([p.x(), p.y()], float)):
                self._giz_dragging = True

    def mouseReleaseEvent(self, e):
        p = e.position().toPoint()
        if getattr(self, "_giz_dragging", False):
            self._giz_dragging = False
            self._gizmo_release()
            return
        # click (khong keo) chuot trai = chon mesh duoi con tro; Ctrl = them vao selection
        if e.button() == Qt.LeftButton and (p - getattr(self, "_press", p)).manhattanLength() < 4 \
                and not (e.modifiers() & Qt.ShiftModifier):
            self.meshPicked.emit(self.pick(p.x(), p.y()), bool(e.modifiers() & Qt.ControlModifier))

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        if getattr(self, "_giz_dragging", False):
            self._last = pos
            self._gizmo_drag(np.array([pos.x(), pos.y()], float))
            return
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        btn = e.buttons()
        alt = e.modifiers() & (Qt.AltModifier | Qt.ShiftModifier)
        if btn & Qt.MiddleButton or (btn & Qt.LeftButton and e.modifiers() & Qt.ShiftModifier):
            v = self.view_matrix()
            right, up = v[0, :3], v[1, :3]
            k = self.dist * 0.0015
            self.target = (self.target - right * dx * k + up * dy * k).astype(np.float32)
        elif btn & Qt.RightButton:
            self.dist *= math.exp(dx * 0.005 - dy * 0.005)
        elif btn & Qt.LeftButton:
            self.yaw -= dx * 0.4
            self.pitch = max(-89.0, min(89.0, self.pitch - dy * 0.4))
        self.update()

    def wheelEvent(self, e):
        self.dist *= 0.88 if e.angleDelta().y() > 0 else 1 / 0.88
        self.update()

    def cycle_background(self):
        self.bg_index = (self.bg_index + 1) % len(self.bg_list)
        self.backgroundChanged.emit(self.bg_list[self.bg_index][2])
        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F:
            self.frameRequested.emit()
        elif e.key() == Qt.Key_B and (e.modifiers() & Qt.ControlModifier):
            self.cycle_background()
        elif e.key() == Qt.Key_Q:
            self.set_gizmo_mode(None)
        elif e.key() == Qt.Key_W:
            self.set_gizmo_mode("move")
        elif e.key() == Qt.Key_E:
            self.set_gizmo_mode("rotate")
        elif e.key() == Qt.Key_R:
            self.set_gizmo_mode("scale")
        else:
            super().keyPressEvent(e)

    # ------------------------------------------------------------ gizmo
    AXCOL = [(0.95, 0.25, 0.30), (0.40, 0.90, 0.35), (0.30, 0.55, 1.0)]

    def set_gizmo_target(self, key):
        self.gizmo_key = key or None
        self._giz_axis = None
        self.update()

    def set_gizmo_mode(self, mode):
        self.gizmo_mode = mode
        self._giz_axis = None
        self.gizmoModeChanged.emit(mode or "")
        self.update()

    def _giz_mesh(self):
        if not self.gizmo_key or not self.gizmo_mode:
            return None
        for m in self.meshes:
            if m.key == self.gizmo_key and m.visible:
                return m
        return None

    def _giz_trs(self, m):
        return m.trs if m.trs else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    def _giz_frame(self, m):
        """Tra ve (center world, [3 truc huong world], do dai world)."""
        t, r, s = self._giz_trs(m)
        c0 = (m.bmin + m.bmax) / 2
        center = c0 + np.asarray(t, np.float64)
        if self.gizmo_mode == "scale":
            R = _euler_xyz(r)
            axes = [R[:, i] for i in range(3)]
        else:
            axes = [np.eye(3)[i] for i in range(3)]
        L = max(1e-6, 0.16 * self.dist)
        return center, axes, L

    def _project(self, p, view, proj, W, H):
        v = np.append(np.asarray(p, np.float64), 1.0)
        clip = proj @ (view @ v)
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        return np.array([(ndc[0] * 0.5 + 0.5) * W, (1 - (ndc[1] * 0.5 + 0.5)) * H])

    def _gizmo_segments(self, m):
        """Danh sach (p0, p1, color) world de ve gizmo theo che do hien tai."""
        center, axes, L = self._giz_frame(m)
        segs = []
        if self.gizmo_mode in ("move", "scale"):
            for i, a in enumerate(axes):
                col = self.AXCOL[i]
                tip = center + a * L
                segs.append((center, tip, col))
                # mui ten / hop o dau truc
                ref = axes[(i + 1) % 3]
                ref2 = axes[(i + 2) % 3]
                if self.gizmo_mode == "move":
                    for sgn in (ref, -ref, ref2, -ref2):
                        segs.append((tip, tip - a * L * 0.16 + sgn * L * 0.07, col))
                else:  # scale: hop nho o dau
                    d = L * 0.05
                    cs = [tip + sx * ref * d + sy * ref2 * d for sx in (-1, 1) for sy in (-1, 1)]
                    segs += [(cs[0], cs[1], col), (cs[1], cs[3], col), (cs[3], cs[2], col), (cs[2], cs[0], col)]
        else:  # rotate: 3 vong tron
            N = 48
            for i, a in enumerate(axes):
                col = self.AXCOL[i]
                u = axes[(i + 1) % 3]
                w = axes[(i + 2) % 3]
                pts = [center + (math.cos(2 * math.pi * k / N) * u + math.sin(2 * math.pi * k / N) * w) * L
                       for k in range(N + 1)]
                for k in range(N):
                    segs.append((pts[k], pts[k + 1], col))
        return segs

    def _draw_gizmo(self, view, proj):
        try:
            self._draw_gizmo_impl(view, proj)
        except Exception:  # noqa
            import traceback
            traceback.print_exc()

    def _draw_gizmo_impl(self, view, proj):
        m = self._giz_mesh()
        if m is None:
            return
        segs = self._gizmo_segments(m)
        data = []
        for p0, p1, c in segs:
            hi = (1.0, 0.9, 0.3)
            col = hi if (self._giz_axis is not None and c == self.AXCOL[self._giz_axis]) else c
            data += [p0[0], p0[1], p0[2], col[0], col[1], col[2],
                     p1[0], p1[1], p1[2], col[0], col[1], col[2]]
        arr = np.array(data, np.float32)
        if self.giz_vao is None:
            self.giz_vao = GL.glGenVertexArrays(1)
            self.giz_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.giz_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.giz_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 24, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 24, ctypes.c_void_p(12))
        GL.glUseProgram(self.lprog)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.lprog, "uView"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.lprog, "uProj"), 1, GL.GL_TRUE, proj)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDrawArrays(GL.GL_LINES, 0, len(data) // 6)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(0)

    @staticmethod
    def _dist_seg(p, a, b):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            return float(np.linalg.norm(p - a))
        t = max(0.0, min(1.0, float((p - a) @ ab) / L2))
        return float(np.linalg.norm(p - (a + ab * t)))

    def _pick_gizmo_axis(self, mouse):
        m = self._giz_mesh()
        if m is None:
            return None
        W, H = max(1, self.width()), max(1, self.height())
        view, proj = self._matrices(W, H)
        center, axes, L = self._giz_frame(m)
        cs = self._project(center, view, proj, W, H)
        if cs is None:
            return None
        thresh = 11.0
        best = (thresh, None)
        if self.gizmo_mode in ("move", "scale"):
            for i, a in enumerate(axes):
                ts = self._project(center + a * L, view, proj, W, H)
                if ts is None:
                    continue
                d = self._dist_seg(mouse, cs, ts)
                if d < best[0]:
                    best = (d, i)
        else:
            N = 36
            for i, a in enumerate(axes):
                u = axes[(i + 1) % 3]
                w = axes[(i + 2) % 3]
                dmin = 1e9
                for k in range(N):
                    ang = 2 * math.pi * k / N
                    ps = self._project(center + (math.cos(ang) * u + math.sin(ang) * w) * L, view, proj, W, H)
                    if ps is not None:
                        dmin = min(dmin, float(np.linalg.norm(mouse - ps)))
                if dmin < best[0]:
                    best = (dmin, i)
        return best[1]

    def _gizmo_press(self, mouse):
        axis = self._pick_gizmo_axis(mouse)
        if axis is None:
            return False
        m = self._giz_mesh()
        t, r, s = self._giz_trs(m)
        W, H = max(1, self.width()), max(1, self.height())
        view, proj = self._matrices(W, H)
        center, axes, L = self._giz_frame(m)
        cs = self._project(center, view, proj, W, H)
        ts = self._project(center + axes[axis] * L, view, proj, W, H)
        eye = self._eye()
        self._giz_axis = axis
        self._giz_snap = dict(t=np.asarray(t, float), r=np.asarray(r, float), s=np.asarray(s, float),
                              center=center, axes=axes, L=L, cs=cs, ts=ts, mouse0=np.asarray(mouse, float),
                              toward=float(axes[axis] @ (eye - center)) >= 0)
        self.transformBegin.emit(m.key, tuple(float(x) for x in t),
                                 tuple(float(x) for x in r), tuple(float(x) for x in s))
        self.update()
        return True

    def _eye(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        d = np.array([math.cos(p) * math.sin(y), math.sin(-p), math.cos(p) * math.cos(y)])
        return self.target + d * self.dist

    def _gizmo_drag(self, mouse):
        snap = self._giz_snap
        if snap is None:
            return
        m = self._giz_mesh()
        if m is None:
            return
        axis = self._giz_axis
        mouse = np.asarray(mouse, float)
        t = snap["t"].copy()
        r = snap["r"].copy()
        s = snap["s"].copy()
        mv = mouse - snap["mouse0"]
        if self.gizmo_mode in ("move", "scale") and snap["cs"] is not None and snap["ts"] is not None:
            seg = snap["ts"] - snap["cs"]
            seglen = float(np.linalg.norm(seg)) or 1.0
            dirhat = seg / seglen
            along = float(mv @ dirhat) / seglen           # don vi = chieu dai truc
            if self.gizmo_mode == "move":
                t = snap["t"] + snap["axes"][axis] * (along * snap["L"])
            else:
                f = max(0.01, 1.0 + along)
                s = snap["s"].copy()
                s[axis] = max(0.01, snap["s"][axis] * f)
        elif self.gizmo_mode == "rotate" and snap["cs"] is not None:
            a0 = snap["mouse0"] - snap["cs"]
            a1 = mouse - snap["cs"]
            ang = math.atan2(a1[1], a1[0]) - math.atan2(a0[1], a0[0])
            # man hinh truc y huong xuong -> dao dau; chieu theo truc huong ve camera
            sgn = -1.0 if snap["toward"] else 1.0
            r = snap["r"].copy()
            r[axis] = snap["r"][axis] + math.degrees(ang) * sgn
        tt = tuple(float(x) for x in t)
        rr = tuple(float(x) for x in r)
        ss = tuple(float(x) for x in s)
        m.set_user_transform(tt, rr, ss)
        self.transformChanged.emit(m.key, tt, rr, ss)
        self.update()

    def _gizmo_release(self):
        m = self._giz_mesh()
        if m is not None:
            t, r, s = self._giz_trs(m)
            self.transformEnd.emit(m.key, tuple(float(x) for x in t),
                                   tuple(float(x) for x in r), tuple(float(x) for x in s))
        self._giz_axis = None
        self._giz_snap = None
        self.update()

    def grab_png(self, path):
        self.grabFramebuffer().save(path)
