# -*- coding: utf-8 -*-
"""
NR Viewer - xem truoc file .nr cua Ninja Ripper 2.x tren Windows.
Python 3 + PySide6 + PyOpenGL + numpy.

Chay:  python nr_viewer.py [duong_dan_thu_muc_frame]
"""
import hashlib
import os
import re
import sys
from collections import Counter

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSettings, QTimer, QSize, QPointF, QRectF
from PySide6.QtGui import (QPixmap, QAction, QKeySequence, QColor, QBrush, QIcon,
                           QPainter, QPen, QPolygonF, QPainterPath)
from PySide6.QtWidgets import (
    QToolButton as _QTB,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QTreeWidget, QTreeWidgetItem,
    QSplitter, QFileDialog, QMessageBox, QScrollArea, QFrame, QToolButton, QSizePolicy,
    QAbstractItemView, QPlainTextEdit, QStatusBar, QProgressBar,
)

import fbx_writer as FBX
import nr_format as NR
import updater as UPD
from gl_view import GLViewport
import gl_view as GLV

APP_NAME = "NR Viewer"
APP_VERSION = "1.0.0"
# === SUA DONG NAY thanh repo GitHub cua ban, dang "tai-khoan/ten-repo" ===
GITHUB_REPO = "3DSixStudio/NR-Viewer"


def resource_path(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def app_icon():
    for ext in (".ico", ".png"):
        p = resource_path("icon" + ext)
        if os.path.isfile(p):
            return QIcon(p)
    return QIcon()

DARK_QSS = """
QWidget { background:#232327; color:#dcdcdc; font-family:'Segoe UI'; font-size:9pt; }
QMainWindow::separator { background:#1a1a1d; width:3px; }
QPushButton, QToolButton { background:#3a3a40; border:1px solid #4a4a52; border-radius:4px; padding:5px 10px; }
QPushButton:hover, QToolButton:hover { background:#4a4a55; }
QPushButton:pressed { background:#2d6cdf; }
QPushButton#accent { background:#2d6cdf; border-color:#3d7cef; color:white; font-weight:bold; }
QPushButton#accent:hover { background:#3d7cef; }
QPushButton:checked { background:#2d6cdf; color:white; border-color:#3d7cef; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
    background:#1b1b1e; border:1px solid #3d3d45; border-radius:3px; padding:3px; }
QTreeWidget { background:#1b1b1e; border:1px solid #333; alternate-background-color:#202024; }
QTreeWidget::item:selected { background:#2d6cdf; color:white; }
QHeaderView::section { background:#2c2c31; border:none; border-right:1px solid #3a3a40; padding:4px; }
QCheckBox::indicator { width:14px; height:14px; }
QScrollArea { border:none; }
QToolButton#sectionHeader { background:#2c2c31; border:none; border-radius:0; text-align:left;
    padding:6px; font-weight:bold; color:#8fb4ff; }
QStatusBar { background:#1a1a1d; }
QProgressBar { border:1px solid #3d3d45; border-radius:3px; text-align:center; height:12px; }
QProgressBar::chunk { background:#2d6cdf; }
"""


def _btn(text, cb=None, accent=False, tip=""):
    b = QPushButton(text)
    if accent:
        b.setObjectName("accent")
    if cb:
        b.clicked.connect(cb)
    if tip:
        b.setToolTip(tip)
    return b


class CollapsibleSection(QWidget):
    def __init__(self, title, expanded=True):
        super().__init__()
        self.head = QToolButton(objectName="sectionHeader")
        self.head.setText(("▼ " if expanded else "► ") + title)
        self.head.setCheckable(True)
        self.head.setChecked(expanded)
        self.head.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.title = title
        self.body = QWidget()
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(8, 6, 8, 8)
        self.lay.setSpacing(5)
        self.body.setVisible(expanded)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.head)
        v.addWidget(self.body)
        self.head.toggled.connect(self._toggle)

    def _toggle(self, on):
        self.body.setVisible(on)
        self.head.setText(("▼ " if on else "► ") + self.title)

    def add(self, w):
        if isinstance(w, QWidget):
            self.lay.addWidget(w)
        else:
            self.lay.addLayout(w)
        return w


def _arrow_head(path, tip, d, w):
    """Mui ten tai tip, huong theo vector d (da chuan hoa), be rong w."""
    import math as _m
    n = (-d[1], d[0])
    base = (tip[0] - d[0] * w * 1.6, tip[1] - d[1] * w * 1.6)
    p = QPolygonF([QPointF(*tip),
                   QPointF(base[0] + n[0] * w, base[1] + n[1] * w),
                   QPointF(base[0] - n[0] * w, base[1] - n[1] * w)])
    path.addPolygon(p)


def tool_icon(name, size=48, color="#d6dae4"):
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    col = QColor(color)
    pen = QPen(col, size * 0.085, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(col)
    c = size / 2
    if name == "select":
        pts = [(0.30, 0.18), (0.30, 0.80), (0.44, 0.66), (0.54, 0.86),
               (0.63, 0.82), (0.53, 0.63), (0.70, 0.60)]
        poly = QPolygonF([QPointF(x * size, y * size) for x, y in pts])
        p.setPen(QPen(col, size * 0.05, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPolygon(poly)
    elif name == "move":
        L = size * 0.33
        hw = size * 0.085
        p.setPen(pen)
        p.drawLine(QPointF(c - L, c), QPointF(c + L, c))
        p.drawLine(QPointF(c, c - L), QPointF(c, c + L))
        path = QPainterPath()
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            tip = (c + d[0] * (L + hw * 0.4), c + d[1] * (L + hw * 0.4))
            _arrow_head(path, tip, d, hw)
        p.setPen(Qt.NoPen)
        p.drawPath(path)
    elif name == "rotate":
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        r = size * 0.30
        rect = QRectF(c - r, c - r, 2 * r, 2 * r)
        p.drawArc(rect, int(60 * 16), int(250 * 16))
        import math as _m
        a = _m.radians(60)
        tip = (c + r * _m.cos(a), c - r * _m.sin(a))
        d = (_m.sin(a), _m.cos(a))  # tiep tuyen
        path = QPainterPath()
        _arrow_head(path, tip, d, size * 0.085)
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawPath(path)
    elif name == "scale":
        p.setPen(pen)
        p.drawLine(QPointF(size * 0.28, size * 0.72), QPointF(size * 0.66, size * 0.34))
        path = QPainterPath()
        import math as _m
        d = (_m.cos(_m.radians(-45)), _m.sin(_m.radians(-45)))
        _arrow_head(path, (size * 0.70, size * 0.30), d, size * 0.085)
        p.setPen(Qt.NoPen)
        p.drawPath(path)
        # o vuong o goc
        p.setPen(QPen(col, size * 0.07))
        p.setBrush(Qt.NoBrush)
        sq = size * 0.17
        p.drawRect(QRectF(size * 0.18, size * 0.60, sq, sq))
    p.end()
    return QIcon(pm)


def _row(*widgets):
    h = QHBoxLayout()
    h.setSpacing(5)
    for w in widgets:
        if isinstance(w, str):
            h.addWidget(QLabel(w))
        elif w is None:
            h.addStretch(1)
        else:
            h.addWidget(w)
    return h


class NumItem(QTreeWidgetItem):
    def __lt__(self, other):
        c = self.treeWidget().sortColumn()
        a, b = self.data(c, Qt.UserRole), other.data(c, Qt.UserRole)
        if a is not None and b is not None:
            return a < b
        return super().__lt__(other)


# --------------------------------------------------------------------------- loader thread
class LoadWorker(QThread):
    meshReady = Signal(int, str, object, object, object, object, object, object)
    progress = Signal(int, int, int)
    failed = Signal(int, str, str)
    duplicate = Signal(int, str, str)

    def __init__(self, gen, jobs, scene, fov, aspect, flip_y, dedupe=True, clip_fns=None, xform=None, vc_index=0):
        super().__init__()
        self.xform = xform
        self.vc_index = vc_index
        self.clip_fns = clip_fns or {}
        self.dedupe = dedupe
        self._seen = {}
        self.gen, self.jobs, self.scene = gen, jobs, scene
        self.fov, self.aspect, self.flip_y = fov, aspect, flip_y
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        n = len(self.jobs)
        cursor = 0.0
        for i, (path, tex, uv_label, dsel, name) in enumerate(self.jobs):
            if self._stop:
                return
            try:
                m = NR.NRMesh(path, draw_sel=dsel)
                if self.scene:
                    cf = self.clip_fns.get(name)
                    g = m.view_geometry(self.fov, self.aspect, uv_label, clip_fn=cf, with_color=True, color_index=self.vc_index)
                    if g is None:
                        continue
                    pos, uv, tri, vcol = g
                    if self.flip_y and cf is None:
                        pos[:, 1] *= -1
                        tri = tri[:, ::-1].copy()
                    if self.xform is not None:          # nan theo skybox
                        pos = (pos.astype(np.float64) @ self.xform.T).astype(np.float32)
                        if np.linalg.det(self.xform) < 0:
                            tri = tri[:, ::-1].copy()
                    off = (0, 0, 0)
                else:
                    uva = None
                    if uv_label and m.input is not None:
                        uva = m.input.find(lambda a: a.label == uv_label)
                    g = m.local_geometry(uva, with_color=True, color_index=self.vc_index)
                    if g is None:
                        continue
                    pos, uv, tri, vcol = g
                    off = (0, 0, 0)
                    if n > 1:  # nhieu mesh -> xep thanh hang ngang
                        mn, mx = pos.min(0), pos.max(0)
                        off = (cursor - mn[0], -mn[1], -(mn[2] + mx[2]) / 2)
                        cursor += (mx[0] - mn[0]) * 1.15 + 1e-3
                if self.scene and self.dedupe:
                    # cung 1 hinh hoc ve lai nhieu lan (depth pre-pass, shadow, outline...)
                    # -> chong khit len nhau gay nhieu "lam tam" (z-fighting)
                    h = hashlib.md5(np.round(pos, 3).tobytes() + tri.tobytes()).hexdigest()
                    if h in self._seen:
                        self.duplicate.emit(self.gen, name, self._seen[h])
                        self.progress.emit(self.gen, i + 1, n)
                        continue
                    self._seen[h] = name
                if not np.isfinite(pos).all():
                    bad = ~np.isfinite(pos).all(1)
                    pos[bad] = 0
                self.meshReady.emit(self.gen, name, pos, uv, tri, tex, np.asarray(off, np.float32), vcol)
            except Exception as e:  # noqa
                self.failed.emit(self.gen, name, str(e))
            self.progress.emit(self.gen, i + 1, n)


# --------------------------------------------------------------------------- main window
class UpdateCheckWorker(QThread):
    done = Signal(object, str)   # info|None, error

    def __init__(self, repo):
        super().__init__()
        self.repo = repo

    def run(self):
        try:
            self.done.emit(UPD.check_latest(self.repo), "")
        except Exception as e:  # noqa
            self.done.emit(None, str(e))


class UpdateDownloadWorker(QThread):
    progress = Signal(int, int)
    done = Signal(str, str)      # file path, error

    def __init__(self, url, dest, expected_size=0):
        super().__init__()
        self.url, self.dest, self.expected = url, dest, expected_size

    def run(self):
        try:
            UPD.download(self.url, self.dest, lambda d, t: self.progress.emit(d, t),
                         expected_size=self.expected)
            self.done.emit(self.dest, "")
        except Exception as e:  # noqa
            self.done.emit("", str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1500, 900)
        self.settings = QSettings("NRViewer", "NRViewer")
        self.folder = ""
        self.meta = {}           # name -> NRMesh (header only)
        self.choice = {}         # name -> [tex_index, uv_label]
        self.tex_freq = Counter()
        self.ripinfo = {}
        self.gen = 0
        self.vc_index = -1         # -1 = Auto: tu do kenh mau that tung mesh (giong Blender)
        self.worker = None
        self._old_workers = []
        self._upd_worker = None
        self._upd_info = None
        self._loaded_keys = []
        self.hidden = set()
        self.trs = {}            # ten mesh -> (t, r, s) nguoi dung dat
        self._undo = []          # ngan xep hoan tac [(key, before, after)]
        self._redo = []
        self._tf_before = None
        self.sky_names = []      # mesh dung lam skybox de nan
        self.sky_M = None
        self.dupes = {}
        self._from_pick = False

        self._build_ui()
        self._sel_timer = QTimer(self, singleShot=True, interval=120, timeout=self._reload_view)

    # ------------------------------------------------------------- UI
    def _build_ui(self):
        split = QSplitter(Qt.Horizontal)
        self.setCentralWidget(split)

        # ---- trai: danh sach
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.addLayout(_row(_btn("📂 Mở thư mục…", self.open_folder, accent=True), None))
        self.lbl_folder = QLabel("Chưa mở thư mục")
        self.lbl_folder.setWordWrap(True)
        self.lbl_folder.setStyleSheet("color:#888")
        lv.addWidget(self.lbl_folder)
        self.ed_search = QLineEdit(placeholderText="Lọc tên / số draw / tên texture…")
        self.ed_search.textChanged.connect(self.apply_filter)
        lv.addWidget(self.ed_search)
        self.cb_hide2d = QCheckBox("Ẩn quad / UI 2D")
        self.cb_hide2d.setChecked(False)   # mac dinh HIEN TAT CA mesh (giong Blender)
        self.cb_hide2d.toggled.connect(self.apply_filter)
        self.sp_mintri = QSpinBox(minimum=0, maximum=10_000_000, value=0)
        self.sp_mintri.valueChanged.connect(self.apply_filter)
        lv.addLayout(_row(self.cb_hide2d, None, "Tris ≥", self.sp_mintri))

        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["Mesh", "Draw", "Đỉnh", "Tam giác", "Inst", "Tex"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setUniformRowHeights(True)
        for i, w in enumerate([110, 50, 70, 80, 40, 35]):
            self.tree.setColumnWidth(i, w)
        self.tree.itemSelectionChanged.connect(self._on_select)
        lv.addWidget(self.tree, 1)
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("color:#888")
        lv.addWidget(self.lbl_count)
        split.addWidget(left)

        # ---- giua: viewport
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 6, 0, 0)
        self.cmb_space = QComboBox()
        self.cmb_space.addItems(["Mesh gốc (object space)", "Cảnh dựng lại (camera game)"])
        self.cmb_space.currentIndexChanged.connect(self._on_space)
        self.cmb_shade = QComboBox()
        self.cmb_shade.addItems(["Texture", "Clay", "Normal", "UV checker", "Vertex Color"])
        self.cmb_shade.currentIndexChanged.connect(self._on_display)
        self.cb_wire = QCheckBox("Wireframe")
        self.cb_wire.toggled.connect(self._on_display)
        self.cb_cull = QCheckBox("Cull mặt sau")
        self.cb_cull.toggled.connect(self._on_display)
        self.cb_grid = QCheckBox("Lưới")
        self.cb_grid.setChecked(True)
        self.cb_grid.toggled.connect(self._on_display)
        mv.addLayout(_row(self.cmb_space, self.cmb_shade, self.cb_wire, self.cb_cull, self.cb_grid, None,
                          _btn("Frame (F)", lambda: self.view.frame_all(set(self.selected_names()) or None)),
                          _btn("Camera gốc", self._camera_origin, tip="Nhìn từ vị trí camera của game")))
        self.view = GLViewport()
        self.view.statsChanged.connect(lambda s: self.lbl_stats.setText(s))
        self.view.meshPicked.connect(self._on_pick)
        self.view.frameRequested.connect(self._frame_selected)
        self.view.backgroundChanged.connect(lambda n: self.statusBar().showMessage(f"Nền viewport: {n}", 2500))
        self.view.transformChanged.connect(self._on_gizmo_transform)
        self.view.transformBegin.connect(self._on_tf_begin)
        self.view.transformEnd.connect(self._on_tf_end)
        self.view.gizmoModeChanged.connect(self._on_gizmo_mode)
        mv.addWidget(self.view, 1)

        # thanh cong cu doc o goc tren trai viewport (kieu Blender)
        self.tool_bar = QWidget(self.view)
        self.tool_bar.setStyleSheet("QWidget{background:rgba(28,30,38,225); border:1px solid #3a3d48; border-radius:9px;}"
                                    "QToolButton{background:transparent; border:none; border-radius:6px; padding:3px;}"
                                    "QToolButton:hover{background:#3a3d4a;}"
                                    "QToolButton:checked{background:#2d6cdf;}")
        tbl = QVBoxLayout(self.tool_bar)
        tbl.setContentsMargins(4, 4, 4, 4)
        tbl.setSpacing(3)
        self.gizmo_btns = {}
        for mode, tip in [(None, "Chọn (Q)"), ("move", "Di chuyển (W)"),
                          ("rotate", "Xoay (E)"), ("scale", "Scale (R)")]:
            b = _QTB()
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setToolTip(tip)
            b.setIcon(tool_icon("select" if mode is None else mode))
            b.setIconSize(QSize(24, 24))
            b.setFixedSize(36, 36)
            b.clicked.connect(lambda _=False, mo=mode: self.view.set_gizmo_mode(mo))
            tbl.addWidget(b)
            self.gizmo_btns[mode] = b
        self.gizmo_btns["move"].setChecked(True)
        self.tool_bar.adjustSize()
        self.tool_bar.move(8, 8)

        # nhan noi tren viewport: ten mesh vua click
        self.lbl_pick = QLabel(self.view)
        self.lbl_pick.setStyleSheet("background:rgba(20,20,24,210); color:#ffb070; border:1px solid #ff8c1a;"
                                    "border-radius:4px; padding:5px 8px; font-size:10pt;")
        self.lbl_pick.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_pick.move(56, 8)
        self.lbl_pick.hide()
        hint = QLabel("Chuột trái: xoay • Giữa/Shift+trái: pan • Lăn/phải: zoom • F: zoom vật chọn • W/E/R: di chuyển/xoay/scale • Ctrl+B: đổi nền")
        hint.setStyleSheet("color:#777; padding:3px")
        mv.addWidget(hint)
        split.addWidget(mid)

        # ---- phai: panel
        right_inner = QWidget()
        rv = QVBoxLayout(right_inner)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(4)

        sec_tex = CollapsibleSection("Texture & UV")
        self.cmb_tex = QComboBox()
        self.cmb_tex.currentIndexChanged.connect(self._on_tex_choice)
        sec_tex.add(QLabel("Texture hiển thị:"))
        sec_tex.add(self.cmb_tex)
        self.lbl_thumb = QLabel()
        self.lbl_thumb.setFixedHeight(200)
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        self.lbl_thumb.setStyleSheet("background:#1b1b1e; border:1px solid #333")
        sec_tex.add(self.lbl_thumb)
        self.cmb_uv = QComboBox()
        self.cmb_uv.currentIndexChanged.connect(self._on_uv_choice)
        sec_tex.add(_row("Kênh UV:", self.cmb_uv))
        self.cmb_vc = QComboBox()
        self.cmb_vc.setToolTip("Kênh vertex color (giống Vertex Color Index của Blender)")
        self.cmb_vc.currentIndexChanged.connect(self._on_vc_choice)
        sec_tex.add(_row("Kênh màu:", self.cmb_vc))
        self.sp_alpha = QDoubleSpinBox(minimum=0, maximum=1, singleStep=0.05, value=0.1)
        self.sp_alpha.valueChanged.connect(self._on_display)
        sec_tex.add(_row("Alpha cutoff:", self.sp_alpha))
        rv.addWidget(sec_tex)

        sec_scene = CollapsibleSection("Cảnh dựng lại")
        self.sp_fov = QDoubleSpinBox(minimum=10, maximum=150, value=70, suffix="°")
        self.sp_fov.setToolTip("FOV dọc của camera game (Roblox mặc định 70°)")
        self.sp_fov.valueChanged.connect(lambda *_: self._sel_timer.start() if self.cmb_space.currentIndex() else None)
        self.cb_flipy = QCheckBox("Lật trục Y (Nox / render ngược)")
        self.cb_flipy.setChecked(True)
        self.cb_flipy.toggled.connect(lambda *_: self._sel_timer.start() if self.cmb_space.currentIndex() else None)
        self.lbl_calib = QLabel("FOV: chưa dò")
        self.lbl_calib.setWordWrap(True)
        self.lbl_calib.setStyleSheet("color:#888")
        sec_scene.add(self.lbl_calib)
        sec_scene.add(_btn("🎯 Tự dò FOV / tỉ lệ (chống méo)", self.calibrate,
                           tip="Tính FOV và tỉ lệ khung thật của camera game từ chính các mesh"))
        self.sp_aspect = QDoubleSpinBox(minimum=0.3, maximum=4, singleStep=0.01, decimals=4, value=16 / 9)
        self.sp_aspect.valueChanged.connect(lambda *_: self._sel_timer.start() if self.cmb_space.currentIndex() else None)
        self.sp_fov.setDecimals(2)
        sec_scene.add(_row("FOV:", self.sp_fov, "Tỉ lệ:", self.sp_aspect))
        sec_scene.add(self.cb_flipy)
        self.cb_dedupe = QCheckBox("Bỏ mesh vẽ trùng (chống nhiễu lấm tấm)")
        self.cb_dedupe.setChecked(False)
        self.cb_dedupe.setToolTip("TẮT (mặc định): giữ TẤT CẢ mesh, không bỏ bản nào.\n"
                                  "BẬT: bỏ bớt các bản vẽ trùng khít (depth pass, shadow) để hết nhiễu z-fighting.")
        self.cb_dedupe.toggled.connect(lambda *_: self._reload_view(self._loaded_keys, False)
                                       if self.cmb_space.currentIndex() else None)
        sec_scene.add(self.cb_dedupe)
        sec_scene.add(_btn("🌐 Dựng cả frame (mesh đang lọc)", self.build_whole_frame, accent=True))
        rv.addWidget(sec_scene)

        sec_sky = CollapsibleSection("Nắn theo mesh chuẩn")
        self.lbl_sky = QLabel("Chưa nắn. Chọn 1 mesh hình đơn giản (skybox, tường, sàn)\n"
                              "rồi bấm nút dưới — cảnh sẽ nắn khớp HÌNH GỐC của mesh đó\n"
                              "(không ép thành lập phương), tự sửa cả lỗi FOV.")
        self.lbl_sky.setWordWrap(True)
        self.lbl_sky.setStyleSheet("color:#888")
        sec_sky.add(self.lbl_sky)
        sec_sky.add(_btn("Dùng mesh đang chọn làm chuẩn nắn", self.skybox_from_selection, accent=True,
                         tip="Chọn 1 mesh hình hộp/phẳng đơn giản rồi bấm"))
        self.cb_sky_align = QCheckBox("Chỉ xoay thẳng, không kéo giãn")
        self.cb_sky_align.setChecked(True)
        self.cb_sky_align.setToolTip("BẬT (mặc định): chỉ xoay cảnh cho thẳng trục theo mesh chuẩn.\n"
                                     "TẮT: xoay + bóp lại tỉ lệ (dùng khi FOV dò sai làm cảnh méo).")
        self.cb_sky_align.toggled.connect(lambda *_: self._apply_sky(self.sky_names) if self.sky_names else None)
        sec_sky.add(self.cb_sky_align)
        sec_sky.add(_btn("Bỏ nắn", self.clear_skybox))
        rv.addWidget(sec_sky)


        sec_exp = CollapsibleSection("Xuất file")
        sec_exp.add(_btn("Xuất FBX – mesh đang chọn", self.export_selected, accent=True))
        sec_exp.add(_btn("Xuất FBX – cảnh (gộp)", self.export_scene, accent=True))
        self.cb_embed = QCheckBox("Nhúng texture vào trong file FBX")
        self.cb_embed.setToolTip("Tắt: texture chép ra thư mục <tên>_textures cạnh file FBX\n"
                                 "Bật: thêm cả ảnh vào trong FBX (file nặng hơn, mang đi 1 file là đủ)")
        sec_exp.add(self.cb_embed)
        self.cb_bake_color = QCheckBox("Tách material theo vertex color (bake)")
        self.cb_bake_color.setChecked(True)
        self.cb_bake_color.setToolTip("Mesh không có texture thật mà có màu đỉnh (Color Attribute)\n"
                                      "sẽ được tách thành từng material theo màu, để Maya nhận màu.")
        sec_exp.add(self.cb_bake_color)
        sec_exp.add(_btn("Lưu ảnh viewport (PNG)", self.save_shot))
        rv.addWidget(sec_exp)

        sec_info = CollapsibleSection("Thông tin")
        self.txt_info = QPlainTextEdit(readOnly=True)
        self.txt_info.setMinimumHeight(260)
        self.txt_info.setStyleSheet("font-family:Consolas; font-size:8pt")
        sec_info.add(self.txt_info)
        rv.addWidget(sec_info)
        rv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(right_inner)
        scroll.setMinimumWidth(290)

        right_col = QWidget()
        rcv = QVBoxLayout(right_col)
        rcv.setContentsMargins(0, 0, 0, 0)
        rcv.setSpacing(0)
        head = QWidget()
        head.setStyleSheet("background:#1a1a1d;")
        hh = QHBoxLayout(head)
        hh.setContentsMargins(6, 4, 6, 4)
        self.lbl_ver = QLabel(f"v{APP_VERSION}")
        self.lbl_ver.setStyleSheet("color:#777")
        self.btn_update = QPushButton("⟳ Kiểm tra cập nhật")
        self.btn_update.clicked.connect(self._on_update_btn)
        hh.addWidget(self.lbl_ver)
        hh.addStretch(1)
        hh.addWidget(self.btn_update)
        rcv.addWidget(head)
        rcv.addWidget(scroll, 1)
        split.addWidget(right_col)
        split.setSizes([400, 820, 300])

        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_stats = QLabel("")
        self.prog = QProgressBar()
        self.prog.setMaximumWidth(220)
        self.prog.setVisible(False)
        sb.addWidget(self.lbl_stats, 1)
        sb.addPermanentWidget(self.prog)

        for sc, fn in [("Ctrl+O", self.open_folder), ("Ctrl+Z", self.undo),
                       ("Ctrl+Y", self.redo), ("Ctrl+Shift+Z", self.redo)]:
            a = QAction(self, triggered=fn)
            a.setShortcut(QKeySequence(sc))
            a.setShortcutContext(Qt.ApplicationShortcut)
            self.addAction(a)

    # ---------------------------------------------------------- folder
    # ---------------------------------------------------------- cap nhat
    def _on_update_btn(self):
        if self._upd_info is not None:       # da co ban moi -> tai va cap nhat
            self._do_update()
        else:
            self._check_update()

    def _check_update(self):
        if "your-name/" in GITHUB_REPO:
            QMessageBox.information(self, APP_NAME,
                                   "Chưa cấu hình repo GitHub.\nMở file nr_viewer.py, sửa dòng GITHUB_REPO "
                                   "thành \"tài-khoản/tên-repo\" của bạn rồi build lại.")
            return
        self.btn_update.setEnabled(False)
        self.btn_update.setText("Đang kiểm tra…")
        self._upd_worker = UpdateCheckWorker(GITHUB_REPO)
        self._upd_worker.done.connect(self._on_check_done)
        self._upd_worker.start()

    def _on_check_done(self, info, err):
        self.btn_update.setEnabled(True)
        if err or not info:
            self.btn_update.setText("⟳ Kiểm tra cập nhật")
            QMessageBox.warning(self, APP_NAME, f"Không kiểm tra được cập nhật:\n{err}")
            return
        if not info.get("version") or not UPD.is_newer(info["version"], APP_VERSION):
            self.btn_update.setText("✓ Đã là bản mới nhất")
            self.statusBar().showMessage(f"Bạn đang dùng bản mới nhất (v{APP_VERSION})", 6000)
            QTimer.singleShot(4000, lambda: self.btn_update.setText("⟳ Kiểm tra cập nhật"))
            return
        self._upd_info = info
        self.btn_update.setObjectName("accent")
        self.btn_update.setStyleSheet("QPushButton{background:#2d6cdf;color:white;font-weight:bold;border-radius:4px;padding:5px 10px;}")
        self.btn_update.setText(f"⬇ Cập nhật {info['version']}")
        self.btn_update.setToolTip((info.get("notes") or "")[:400])

    def _do_update(self):
        info = self._upd_info
        url = info.get("asset_url")
        if not url:
            # khong co file dinh kem -> mo trang release
            import webbrowser
            webbrowser.open(info.get("page", f"https://github.com/{GITHUB_REPO}/releases"))
            return
        if not UPD.is_frozen() or os.name != "nt":
            import webbrowser
            webbrowser.open(info.get("page", f"https://github.com/{GITHUB_REPO}/releases"))
            QMessageBox.information(self, APP_NAME,
                                   "Đang chạy từ mã nguồn nên không tự thay file .exe được.\n"
                                   "Đã mở trang tải bản mới trên trình duyệt.")
            return
        folder = os.path.dirname(UPD.current_exe())
        dest = os.path.join(folder, "NRViewer_new.exe" if url.lower().endswith(".exe") else "NRViewer_update.zip")
        self.btn_update.setEnabled(False)
        self.prog.setVisible(True)
        self.prog.setRange(0, 100)
        self._dl = UpdateDownloadWorker(url, dest, info.get("asset_size", 0))
        self._dl.progress.connect(lambda d, t: self.prog.setValue(int(d * 100 / t)) if t else None)
        self._dl.done.connect(self._on_download_done)
        self._dl.start()
        self.statusBar().showMessage("Đang tải bản cập nhật…")

    def _on_download_done(self, path, err):
        self.prog.setVisible(False)
        self.btn_update.setEnabled(True)
        if err or not path:
            QMessageBox.warning(self, APP_NAME, f"Tải cập nhật thất bại:\n{err}")
            return
        if path.lower().endswith(".zip"):
            import webbrowser
            webbrowser.open(os.path.dirname(path))
            QMessageBox.information(self, APP_NAME,
                                   f"Đã tải bản mới (dạng .zip) về:\n{path}\nHãy giải nén đè để cập nhật.")
            return
        r = QMessageBox.question(self, APP_NAME,
                                 f"Đã tải xong {self._upd_info['version']}.\n"
                                 "Ứng dụng sẽ đóng lại và tự cập nhật rồi mở lại. Tiếp tục?")
        if r != QMessageBox.Yes:
            return
        try:
            UPD.apply_update_and_restart(path)
        except Exception as e:  # noqa
            QMessageBox.warning(self, APP_NAME, f"Không áp dụng được cập nhật:\n{e}")
            return
        QApplication.quit()

    def open_folder(self):
        start = self.settings.value("last_folder", "")
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục frame của Ninja Ripper", start)
        if d:
            self.load_folder(d)

    def load_folder(self, d):
        files = [f for f in os.listdir(d) if f.lower().endswith(".nr")]
        if not files:
            # cho phep chon thu muc cha chua frame_N
            subs = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)) and x.startswith("frame_"))
            if subs:
                return self.load_folder(os.path.join(d, subs[0]))
            QMessageBox.warning(self, APP_NAME, "Không tìm thấy file .nr trong thư mục này.")
            return
        self.settings.setValue("last_folder", d)
        self.folder = d
        self.ripinfo = NR.read_ripdesc(d)
        files.sort(key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)])
        self.meta.clear()
        self.choice.clear()
        self.sky_names, self.sky_M = [], None
        self.tex_freq = Counter()
        self.prog.setVisible(True)
        self.prog.setRange(0, len(files))
        for i, f in enumerate(files):
            try:
                m0 = NR.NRMesh(os.path.join(d, f), header_only=True)
                base = m0.name
                # Y het io_import_nr: MOI GEOM (draw) la 1 object, ten = ten file;
                # trung ten thi them .001, .002... (so trong nho nhat, nhu Blender)
                for k in range(m0.num_draws):
                    m = m0 if k == 0 else NR.NRMesh(os.path.join(d, f), header_only=True, draw_sel=k)
                    m.draw_sel = k
                    nm, n = base, 0
                    while nm in self.meta:
                        n += 1
                        nm = f"{base}.{n:03d}"
                    m.name = nm
                    self.meta[nm] = m
                    self.tex_freq.update(t for _, t in m.textures)
            except Exception as e:  # noqa
                print("Loi doc", f, e)
            if i % 50 == 0:
                self.prog.setValue(i)
                QApplication.processEvents()
        self.prog.setVisible(False)
        for name, m in self.meta.items():
            self.choice[name] = [self._default_tex(m), self._default_uv(m)]
        self._populate_vc()

        info = self.ripinfo
        txt = os.path.normpath(d)
        if info:
            txt += f"\n{info.get('executable', '')}  •  {info.get('width', '?')}×{info.get('height', '?')}"
        self.lbl_folder.setText(txt)
        self.setWindowTitle(f"{APP_NAME} — {os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}")
        self._fill_tree()
        w = int(self.ripinfo.get("width", 16) or 16)
        h = int(self.ripinfo.get("height", 9) or 9)
        self.sp_aspect.blockSignals(True)
        self.sp_aspect.setValue(w / h)
        self.sp_aspect.blockSignals(False)
        QTimer.singleShot(0, self.calibrate)

    # ---------------------------------------------------------- skybox
    def _scene_params(self):
        return self.sp_fov.value(), self.sp_aspect.value(), self.cb_flipy.isChecked()

    def auto_skybox(self):
        if not self.meta:
            return
        fov, asp, fy = self._scene_params()
        cands = [m for m in self.meta.values() if 2 <= m.tri_count() <= 200 and not self._is_2d(m)]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        scored = []
        try:
            for m in cands:
                try:
                    sc = NR.skybox_score(NR.NRMesh(m.path, draw_sel=m.draw_sel), fov, asp, fy)
                except Exception:  # noqa
                    continue
                if sc > 0:
                    scored.append((sc, m.name))
        finally:
            QApplication.restoreOverrideCursor()
        scored.sort(reverse=True)
        # cac mat skybox nam xa nhat va xa xap xi nhau
        pick = [n for sc, n in scored[:6] if scored and sc >= scored[0][0] * 0.6]
        if not pick or not self._apply_sky(pick, quiet=True):
            QMessageBox.information(self, APP_NAME, "Không tự tìm được skybox.\n"
                                    "Ctrl+click chọn 2–3 mặt skybox trong viewport rồi bấm\n"
                                    "\"Dùng mesh đang chọn làm chuẩn nắn\".")

    def skybox_from_selection(self):
        names = self.selected_names()
        if not names:
            QMessageBox.information(self, APP_NAME, "Ctrl+click chọn 2–3 mặt skybox trong viewport trước.")
            return
        self._apply_sky(names)

    def _apply_sky(self, names, reload=True, quiet=False):
        fov, asp, fy = self._scene_params()
        rot_only = self.cb_sky_align.isChecked()
        S_all, O_all, normals = [], [], []
        for n in names:
            m = NR.NRMesh(self.meta[n].path, draw_sel=self.meta[n].draw_sel)
            gv = m.view_geometry(fov, asp)
            gl = m.local_geometry()
            if gv is None or gl is None:
                continue
            ps = gv[0].copy()
            if fy:
                ps[:, 1] *= -1
            nrm = NR.face_normal_scene(ps, gv[2])      # phap tuyen mat trong canh
            if nrm is not None:
                normals.append(nrm)
            po = gl[0]
            k = min(len(ps), len(po))
            if k < 8 or abs(len(ps) - len(po)) > max(len(ps), len(po)) * 0.5:
                continue
            S_all.append(ps[:k])
            O_all.append(po[:k])

        # "Chỉ xoay thẳng" = xoay cho PHÁP TUYẾN các mặt đã chọn vuông góc trục
        # (không phụ thuộc object-space, không méo). Cần >=2 mặt không song song.
        M = None
        if rot_only and len(normals) >= 2:
            M = NR.axis_align_rotation(normals)
        if M is None:
            if not S_all:
                if not quiet:
                    QMessageBox.warning(self, APP_NAME,
                                        "Mesh chọn không nắn được. Hãy chọn 2–3 mặt phẳng "
                                        "(tường/sàn/trần skybox) vuông góc nhau.")
                return False
            _M, info = NR.reference_correction(np.vstack(S_all), np.vstack(O_all), rotate_only=rot_only)
            M = _M
            note = ("chỉ xoay thẳng (dự phòng)" if rot_only
                    else f"xoay + sửa méo: {info['stretch'][0]:.3f}/{info['stretch'][1]:.3f}/{info['stretch'][2]:.3f}")
        else:
            note = f"xoay thẳng theo {len(normals)} mặt (vuông góc trục)"
        self.sky_names, self.sky_M = list(names), M
        self.lbl_sky.setText(f"✓ Nắn theo {', '.join(names)}\n{note}")
        self.lbl_sky.setStyleSheet("color:#7fd07f")
        if reload:
            if self.cmb_space.currentIndex() != 1 or not self._loaded_keys:
                self.build_whole_frame()
            else:
                self._reload_view(self._loaded_keys)
        return True

    def clear_skybox(self):
        self.sky_names, self.sky_M = [], None
        self.lbl_sky.setText("Chưa nắn.")
        self.lbl_sky.setStyleSheet("color:#888")
        if self.cmb_space.currentIndex() == 1 and self._loaded_keys:
            self._reload_view(self._loaded_keys)

    # ---------------------------------------------------------- calibrate
    def calibrate(self):
        """Do FOV + ti le that tu du lieu mesh (khong can file log)."""
        if not self.meta:
            return
        cands = [m for m in self.meta.values()
                 if not self._is_2d(m) and 50 <= m.tri_count() <= 30000 and m.outputs]
        cands.sort(key=lambda m: -m.tri_count())
        self.statusBar().showMessage("Đang dò FOV…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            full = []
            for m in cands[:80]:
                try:
                    full.append(NR.NRMesh(m.path, draw_sel=m.draw_sel))
                except Exception:  # noqa
                    pass
            r = NR.calibrate_projection(full)
        finally:
            QApplication.restoreOverrideCursor()
        if r is None or r[2] > 0.02:
            self.lbl_calib.setText("Không dò được FOV (ít mesh cứng) → dùng giá trị nhập tay")
            self.lbl_calib.setStyleSheet("color:#d0a050")
            self.statusBar().clearMessage()
            return
        fov, aspect, err, n = r
        for w, v in ((self.sp_fov, fov), (self.sp_aspect, aspect)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.lbl_calib.setText(f"✓ Đã dò từ {n} mesh: FOV {fov:.2f}° • tỉ lệ {aspect:.4f} (sai số {err:.1e})")
        self.lbl_calib.setStyleSheet("color:#7fd07f")
        self.statusBar().showMessage(f"FOV camera game ≈ {fov:.2f}°, tỉ lệ {aspect:.3f}", 6000)
        if self.cmb_space.currentIndex() == 1 and self._loaded_keys:
            self._reload_view(self._loaded_keys, False)

    def _default_tex(self, m):
        if not m.textures:
            return -1
        best, bestc = 0, None
        for i, (_, t) in enumerate(m.textures):
            c = self.tex_freq[t]
            if not os.path.isfile(os.path.join(self.folder, t)):
                continue
            if bestc is None or c < bestc:
                best, bestc = i, c
        return best

    def _default_uv(self, m):
        c = m.uv_candidates()
        pa = m.position_attr()
        c = [a for a in c if pa is None or a.label != pa.label]
        return c[0].label if c else None

    def _is_2d(self, m):
        pa = m.position_attr()
        return m.is_screen_quad() or pa is None or pa.comps < 3

    def _fill_tree(self):
        self.tree.setSortingEnabled(False)
        self.tree.clear()
        items = []
        for name, m in self.meta.items():
            it = NumItem()
            vals = [name, m.draw_index, m.vert_count(), m.tri_count(), m.instances, len(m.textures)]
            _t = name.split("_")[-1].split(".")
            _k = (int(_t[0]) if _t[0].isdigit() else 0) + (int(_t[1]) / 1000.0 if len(_t) > 1 and _t[1].isdigit() else 0)
            sort = [_k] + vals[1:]
            for c, (v, s) in enumerate(zip(vals, sort)):
                it.setText(c, f"{v:,}" if isinstance(v, int) and c else str(v))
                it.setData(c, Qt.UserRole, s)
                if c:
                    it.setTextAlignment(c, Qt.AlignRight | Qt.AlignVCenter)
            it.setData(0, Qt.UserRole + 1, name)
            items.append(it)
        self.tree.addTopLevelItems(items)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)
        self.apply_filter()

    def apply_filter(self):
        q = self.ed_search.text().strip().lower()
        # Chế độ dựng cảnh LUÔN ẩn 2D; chế độ Mesh gốc theo ô tích (mặc định hiện hết).
        hide2d = self.cb_hide2d.isChecked() or self.cmb_space.currentIndex() == 1
        mintri = self.sp_mintri.value()
        shown = 0
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            m = self.meta[it.data(0, Qt.UserRole + 1)]
            ok = True
            if hide2d and self._is_2d(m):
                ok = False
            elif m.tri_count() < mintri:
                ok = False
            elif q:
                hay = f"{m.name} {m.draw_index} " + " ".join(t for _, t in m.textures).lower()
                ok = all(tok in hay for tok in q.split())
            it.setHidden(not ok)
            shown += ok
        self.lbl_count.setText(f"Hiển thị {shown} / {len(self.meta)} mesh")

    def visible_names(self):
        out = []
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if not it.isHidden():
                out.append(it.data(0, Qt.UserRole + 1))
        return out

    def selected_names(self):
        return [it.data(0, Qt.UserRole + 1) for it in self.tree.selectedItems()]

    # ---------------------------------------------------------- selection
    def _on_select(self):
        names = self.selected_names()
        self._update_panel(names)
        if self.cmb_space.currentIndex() == 1:
            # Cảnh dựng lại: KHÔNG bao giờ nạp lại viewport.
            # Chọn vật -> tô sáng + zoom tới. Bỏ chọn (click vùng trống) -> chỉ bỏ tô sáng.
            self._sel_timer.stop()
            self.view.set_tint(names)
            if names:
                self.view.frame_all(set(names))
            return
        # Chế độ "Mesh gốc": hiển thị từng mesh được chọn.
        if not names:
            return
        loaded = set(self.view.loaded_keys()) | set(self._loaded_keys)
        subset = set(names) <= loaded and len(loaded) > 1
        if subset and self._from_pick:
            self._sel_timer.stop()
            self.view.set_tint(names)
        else:
            self._sel_timer.start()

    def _find_item(self, name):
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.UserRole + 1) == name:
                return it
        return None

    def _on_pick(self, name, add):
        self._from_pick = True
        try:
            if not name:
                if not add:
                    self.lbl_pick.hide()
                    self.tree.clearSelection()
                    self.view.set_tint([])
                return
            it = self._find_item(name)
            if it is None:
                return
            if it.isHidden():
                it.setHidden(False)
            if add:
                it.setSelected(not it.isSelected())
            else:
                self.tree.setCurrentItem(it)
            self.tree.scrollToItem(it)
            m = self.meta[name]
            tex = self._tex_path(name)
            self.lbl_pick.setText(f"<b>{name}</b> &nbsp; draw #{m.draw_index}<br>"
                                  f"{m.vert_count():,} đỉnh • {m.tri_count():,} tam giác • inst {m.instances}<br>"
                                  f"<span style='color:#aaa'>tex: {os.path.basename(tex) if tex else '(không)'}</span>")
            self.lbl_pick.adjustSize()
            self.lbl_pick.show()
            self.statusBar().showMessage(f"Chọn {name}", 3000)
        finally:
            self._from_pick = False

    # ---------------------------------------------------------- hide
    def _refresh_hidden(self):
        self.view.set_hidden(self.hidden)
        grey, normal = QBrush(QColor("#666")), QBrush()
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            n = it.data(0, Qt.UserRole + 1)
            br = grey if (n in self.hidden or n in self.dupes) else normal
            for c in range(self.tree.columnCount()):
                it.setForeground(c, br)
            it.setToolTip(0, f"Trùng với {self.dupes[n]} (đã bỏ)" if n in self.dupes else
                          ("Đang ẩn" if n in self.hidden else ""))

    # ---------------------------------------------------------- transform (gizmo)
    def _on_gizmo_mode(self, mode):
        for mo, b in self.gizmo_btns.items():
            b.blockSignals(True)
            b.setChecked((mo or "") == (mode or ""))
            b.blockSignals(False)

    @staticmethod
    def _norm_trs(v):
        return tuple(tuple(round(float(x), 6) for x in part) for part in v)

    def _store_trs(self, key, t, r, s):
        if t == (0, 0, 0) and r == (0, 0, 0) and s == (1, 1, 1):
            self.trs.pop(key, None)
        else:
            self.trs[key] = (t, r, s)

    def _on_gizmo_transform(self, key, t, r, s):
        self._store_trs(key, t, r, s)

    def _on_tf_begin(self, key, t, r, s):
        self._tf_before = (key, (t, r, s))

    def _on_tf_end(self, key, t, r, s):
        if self._tf_before is None or self._tf_before[0] != key:
            self._tf_before = None
            return
        before = self._tf_before[1]
        after = (t, r, s)
        self._tf_before = None
        if self._norm_trs(before) != self._norm_trs(after):
            self._undo.append((key, before, after))
            self._redo.clear()

    def _apply_trs(self, key, trs):
        # chi doi bien doi cua mesh, KHONG dong toi camera hay lua chon
        t, r, s = trs
        self._store_trs(key, t, r, s)
        self.view.set_transform(key, t, r, s)

    def undo(self):
        if not self._undo:
            self.statusBar().showMessage("Không còn gì để hoàn tác", 2000)
            return
        key, before, after = self._undo.pop()
        self._apply_trs(key, before)
        self._redo.append((key, before, after))
        self.statusBar().showMessage(f"Hoàn tác biến đổi {key}", 2500)

    def redo(self):
        if not self._redo:
            return
        key, before, after = self._redo.pop()
        self._apply_trs(key, after)
        self._undo.append((key, before, after))
        self.statusBar().showMessage(f"Làm lại biến đổi {key}", 2500)

    def _frame_selected(self):
        names = set(self.selected_names())
        self.view.frame_all(names or None)

    def _update_panel(self, names):
        for w in (self.cmb_tex, self.cmb_uv):
            w.blockSignals(True)
            w.clear()
        if len(names) != 1:
            self.cmb_tex.setEnabled(False)
            self.cmb_uv.setEnabled(False)
            self.lbl_thumb.clear()
            self.txt_info.setPlainText(f"{len(names)} mesh được chọn" if names else "")
            for w in (self.cmb_tex, self.cmb_uv):
                w.blockSignals(False)
            self.view.set_gizmo_target(None)      # gizmo chi hien khi chon dung 1 vat
            return
        m = self.meta[names[0]]
        ch = self.choice[m.name]
        self.cmb_tex.setEnabled(True)
        self.cmb_uv.setEnabled(True)
        self.cmb_tex.addItem("(không texture)", -1)
        for i, (_, t) in enumerate(m.textures):
            exists = os.path.isfile(os.path.join(self.folder, t))
            self.cmb_tex.addItem(f"[{i}] {t}{'' if exists else '  (thiếu)'}  ×{self.tex_freq[t]}", i)
        self.cmb_tex.setCurrentIndex(ch[0] + 1)
        for a in m.uv_candidates():
            self.cmb_uv.addItem(f"{a.label} ({'float' if a.fmt == 0 else 'half'}2)", a.label)
        k = self.cmb_uv.findData(ch[1])
        if k >= 0:
            self.cmb_uv.setCurrentIndex(k)
        for w in (self.cmb_tex, self.cmb_uv):
            w.blockSignals(False)
        self._show_thumb(m, ch[0])
        self.txt_info.setPlainText(self._info_text(m))
        self.view.set_gizmo_target(m.name)

    def _show_thumb(self, m, ti):
        if ti < 0 or ti >= len(m.textures):
            self.lbl_thumb.clear()
            return
        p = os.path.join(self.folder, m.textures[ti][1])
        pm = QPixmap(p)
        if pm.isNull():
            self.lbl_thumb.setText("(không mở được)")
        else:
            self.lbl_thumb.setPixmap(pm.scaled(self.lbl_thumb.width() - 4, 196, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.lbl_thumb.setToolTip(f"{m.textures[ti][1]}\n{pm.width()}×{pm.height()}")

    def _info_text(self, m):
        L = [f"{m.name}   draw #{m.draw_index}   instances: {m.instances}", ""]
        fmtname = {0: "f32", 1: "u32", 2: "u16", 3: "u8", 4: "s32", 5: "s16", 6: "s8"}
        pre, vs = m.draws[m.active] if m.draws else (None, None)
        for bid, lab in ((pre, "PreVs (toạ độ gốc)"), (vs, "Vs / PostVs (màu, dựng cảnh)")):
            if bid is None:
                continue
            b = m.buffers[bid]
            L.append(f"[{lab}]")
            L.append(f"  verts {b.vcount:,}  stride {b.stride}  indices {b.index_count:,}")
            for a in b.attrs:
                L.append(f"   {a.label:<13} {fmtname.get(a.fmt, a.fmt)}x{a.comps}  @{a.offset}")
        L.append("")
        L.append("Textures:")
        for i, (_, t) in enumerate(m.textures):
            L.append(f"  [{i}] {t}")
        L.append("Shaders:")
        for s in m.shaders:
            L.append(f"  {s}")
        return "\n".join(L)

    def _on_tex_choice(self, idx):
        names = self.selected_names()
        if len(names) != 1:
            return
        ti = self.cmb_tex.itemData(idx)
        self.choice[names[0]][0] = ti
        self._show_thumb(self.meta[names[0]], ti)
        self._sel_timer.start()

    def _populate_vc(self):
        """O chon kenh mau chung cho ca canh: vc_0..vc_N (giong Blender)."""
        self.cmb_vc.blockSignals(True)
        self.cmb_vc.clear()
        nmax = NR.max_color_channels(list(self.meta.values()))
        # "Auto" = tu do kenh mau that tung mesh (giong Base Color cua Blender):
        # moi mesh cat mau o so vc khac nhau, nen chon tay 1 so khong dung duoc het.
        self.cmb_vc.addItem("Auto (giống Blender)", -1)
        for i in range(nmax):
            self.cmb_vc.addItem(f"vc_{i}", i)
        pos = self.cmb_vc.findData(self.vc_index)
        self.cmb_vc.setCurrentIndex(pos if pos >= 0 else 0)
        self.cmb_vc.blockSignals(False)

    def _on_vc_choice(self, idx):
        self.vc_index = self.cmb_vc.itemData(idx) or 0
        if self._loaded_keys:
            self._reload_view(self._loaded_keys)

    def _on_uv_choice(self, idx):
        names = self.selected_names()
        if len(names) != 1:
            return
        self.choice[names[0]][1] = self.cmb_uv.itemData(idx)
        self._sel_timer.start()

    # ---------------------------------------------------------- loading
    def _tex_path(self, name):
        m = self.meta[name]
        ti = self.choice[name][0]
        if ti < 0 or ti >= len(m.textures):
            return None
        p = os.path.join(self.folder, m.textures[ti][1])
        return p if os.path.isfile(p) else None

    def _reload_view(self, names=None, frame=True):
        if names is None:
            names = self.selected_names()
        self._start_load(names, frame)

    def _start_load(self, names, frame=True):
        if self.worker is not None:
            self.worker.stop()
            # giu tham chieu toi khi thread cu ket thuc (tranh crash "QThread destroyed")
            old = self.worker
            self._old_workers.append(old)
            old.finished.connect(lambda o=old: self._old_workers.remove(o) if o in self._old_workers else None)
        self.gen += 1
        self.view.clear()
        scene = self.cmb_space.currentIndex() == 1
        if scene:
            # Cảnh dựng lại: LUÔN bỏ quad UI / mesh 2D (không dựng được vào
            # không gian camera -> nếu để vào sẽ thành tam giác khổng lồ, loạn).
            # "Hiện tất cả mesh" chỉ áp cho chế độ Mesh gốc (object space).
            names = [n for n in names if not self._is_2d(self.meta[n])]
        self._loaded_keys = list(names)
        self.dupes = {}
        if not names:
            return
        jobs = [(self.meta[n].path, self._tex_path(n), self.choice[n][1], self.meta[n].draw_sel, n) for n in names]
        if scene and self.sky_names:
            self._apply_sky(self.sky_names, reload=False)
        self.worker = LoadWorker(self.gen, jobs, scene, self.sp_fov.value(), self.sp_aspect.value(), self.cb_flipy.isChecked(),
                                 self.cb_dedupe.isChecked(),
                                 None, self.sky_M if scene else None, self.vc_index)
        self.worker.duplicate.connect(self._on_dupe)
        self.worker.meshReady.connect(self._on_mesh_ready)
        self.worker.progress.connect(self._on_progress)
        self.worker.failed.connect(lambda g, n, e: self.statusBar().showMessage(f"Lỗi {n}: {e}", 5000))
        self._frame_after = frame
        self._first_mesh = True
        self.prog.setVisible(len(jobs) > 3)
        self.prog.setRange(0, len(jobs))
        self.worker.start()

    def _on_dupe(self, gen, name, orig):
        if gen == self.gen:
            self.dupes[name] = orig

    def _on_mesh_ready(self, gen, name, pos, uv, tri, tex, off, vcol=None):
        if gen != self.gen:
            return
        self.view.add_mesh(name, pos, uv, tri, tex, off, vcol)
        if name in self.trs:
            t, r, sc = self.trs[name]
            QTimer.singleShot(0, lambda n=name, t=t, r=r, sc=sc: self.view.set_transform(n, t, r, sc))
        if self._first_mesh and self._frame_after:
            self._first_mesh = False
            QTimer.singleShot(0, self._auto_frame)

    def _auto_frame(self):
        self.view.update()
        if self.cmb_space.currentIndex() == 1:
            QTimer.singleShot(30, self._camera_origin)
        else:
            QTimer.singleShot(30, self.view.frame_all)

    def _on_progress(self, gen, i, n):
        if gen != self.gen:
            return
        self.prog.setValue(i)
        if i >= n:
            self.prog.setVisible(False)
            self._refresh_hidden()
            if self.dupes:
                self.statusBar().showMessage(f"Đã bỏ {len(self.dupes)} mesh vẽ trùng", 6000)
            if self._frame_after:
                QTimer.singleShot(50, self._auto_frame)

    def build_whole_frame(self):
        if not self.meta:
            return
        if self.cmb_space.currentIndex() != 1:
            self.cmb_space.blockSignals(True)
            self.cmb_space.setCurrentIndex(1)
            self.cmb_space.blockSignals(False)
            self.view.show_grid = False
            self.cb_grid.setChecked(False)
        self._start_load(self.visible_names())

    # ---------------------------------------------------------- display
    def _on_space(self, idx):
        self.cb_grid.setChecked(idx == 0)
        self.apply_filter()   # cap nhat danh sach (scene -> an 2D, object -> hien het)
        self._reload_view(self._loaded_keys or self.selected_names())

    def _on_display(self, *_):
        v = self.view
        v.mode = self.cmb_shade.currentIndex()
        v.wire = self.cb_wire.isChecked()
        v.cull = self.cb_cull.isChecked()
        v.show_grid = self.cb_grid.isChecked()
        v.alpha_cut = self.sp_alpha.value()
        v.update()

    def _camera_origin(self):
        if self.cmb_space.currentIndex() != 1:
            self.cmb_space.setCurrentIndex(1)
        if self.sky_M is not None:
            fwd = self.sky_M @ np.array([0.0, 0.0, -1.0])
            self.view.set_camera(np.zeros(3), fwd, self.sp_fov.value())
        else:
            self.view.camera_origin(self.sp_fov.value())

    # ---------------------------------------------------------- export
    def _is_real_texture(self, path):
        """Texture that (co kich thuoc > 4px), khac voi o mau 1x1 cua part mau tron."""
        if not path:
            return False
        sz = NR.png_size(path)
        return sz is not None and (sz[0] > 4 or sz[1] > 4)

    def _build_item(self, name, pos, uv, tri, col):
        """Tao mesh dict: bake vertex color -> material neu khong co texture that."""
        tex = self._tex_path(name)
        if self.cb_bake_color.isChecked() and col is not None and not self._is_real_texture(tex):
            pal, fm = NR.split_faces_by_color(col, tri)
            return dict(name=name, pos=pos, uv=None, tri=tri, colors=pal, face_mat=fm), len(pal)
        return dict(name=name, pos=pos, uv=uv, tri=tri, texture=tex), 0

    def _write_fbx(self, path, items, big_color=None):
        try:
            r = FBX.write_fbx(path, items, embed_textures=self.cb_embed.isChecked(), copy_textures=True)
        except Exception as e:  # noqa
            QMessageBox.warning(self, APP_NAME, f"Lỗi ghi FBX:\n{e}")
            return
        msg = f"Đã xuất {r['models']} mesh, {r['materials']} material, {r['textures']} texture → {path}"
        self.statusBar().showMessage(msg, 10000)
        extra = ""
        if r["tex_dir"]:
            extra += f"\n\nTexture đã chép vào:\n{r['tex_dir']}"
        if big_color:
            lst = ", ".join(f"{n} ({c})" for n, c in big_color[:6])
            extra += ("\n\n⚠ Một số mesh ra rất nhiều màu (có thể do game nướng sẵn đổ bóng vào "
                      f"vertex color):\n{lst}\nNhững mesh này tạo nhiều material.")
        QMessageBox.information(self, APP_NAME, msg + extra)

    def export_selected(self):
        names = [n for n in self.selected_names() if n not in self.dupes]
        if not names:
            QMessageBox.information(self, APP_NAME, "Chọn ít nhất 1 mesh trong danh sách.")
            return
        default = names[0] if len(names) == 1 else "meshes"
        p, _ = QFileDialog.getSaveFileName(self, "Xuất FBX", os.path.join(self.folder, default + ".fbx"), "FBX (*.fbx)")
        if not p:
            return
        items, big = [], []
        for n in names:
            try:
                m = NR.NRMesh(self.meta[n].path, draw_sel=self.meta[n].draw_sel)
                uva = m.input.find(lambda a: a.label == self.choice[n][1]) if m.input else None
                g = m.local_geometry(uva, with_color=True, color_index=self.vc_index)
                if g is None:
                    continue
                pos, uv, tri, col = g
                if n in self.trs:
                    pos = GLV.apply_trs(pos, *self.trs[n])
                item, ncol = self._build_item(n, pos, uv, tri, col)
                items.append(item)
                if ncol > 200:
                    big.append((n, ncol))
            except Exception as e:  # noqa
                print(n, e)
        self._write_fbx(p, items, big)

    def export_scene(self):
        names = self._loaded_keys if self.cmb_space.currentIndex() == 1 else self.selected_names()
        names = [n for n in names if n not in self.hidden and n not in self.dupes]
        if not names:
            QMessageBox.information(self, APP_NAME, "Hãy dựng cảnh hoặc chọn mesh trước.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Xuất FBX cảnh", os.path.join(self.folder, "scene.fbx"), "FBX (*.fbx)")
        if not p:
            return
        items, big = [], []
        self.prog.setVisible(True)
        self.prog.setRange(0, len(names))
        for i, n in enumerate(names):
            try:
                m = NR.NRMesh(self.meta[n].path, draw_sel=self.meta[n].draw_sel)
                g = m.view_geometry(self.sp_fov.value(), self.sp_aspect.value(), self.choice[n][1], with_color=True,
                                    color_index=self.vc_index)
                if g is None:
                    continue
                pos, uv, tri, col = g
                if self.cb_flipy.isChecked():
                    pos[:, 1] *= -1
                    tri = tri[:, ::-1]
                if self.sky_M is not None:
                    pos = pos.astype(np.float64) @ self.sky_M.T
                    if np.linalg.det(self.sky_M) < 0:
                        tri = tri[:, ::-1]
                if n in self.trs:
                    pos = GLV.apply_trs(pos, *self.trs[n])
                item, ncol = self._build_item(n, pos, uv, tri, col)
                items.append(item)
                if ncol > 200:
                    big.append((n, ncol))
            except Exception as e:  # noqa
                print(n, e)
            self.prog.setValue(i + 1)
            QApplication.processEvents()
        self.prog.setVisible(False)
        self._write_fbx(p, items, big)

    def save_shot(self):
        p, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", os.path.join(self.folder or "", "viewport.png"), "PNG (*.png)")
        if p:
            self.view.grab_png(p)


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    app.setStyleSheet(DARK_QSS)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        QTimer.singleShot(100, lambda: w.load_folder(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
