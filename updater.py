# -*- coding: utf-8 -*-
"""
Cap nhat online cho NR Viewer qua GitHub Releases.

- check_latest(repo): doc ban moi nhat tu API GitHub.
- is_newer(remote, local): so sanh phien ban kieu 1.2.0.
- download(url, dest, cb): tai file, bao tien do.
- apply_update_and_restart(new_exe): thay file .exe dang chay roi mo lai (Windows).

Khong can thu vien ngoai, chi dung urllib trong chuan Python.
"""
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.request

API = "https://api.github.com/repos/{repo}/releases/latest"


def parse_version(s):
    s = re.sub(r"[^0-9.]", "", str(s).lstrip("vV"))
    parts = [int(x) for x in s.split(".") if x != ""]
    return tuple(parts) if parts else (0,)


def is_newer(remote, local):
    a, b = parse_version(remote), parse_version(local)
    n = max(len(a), len(b))
    a += (0,) * (n - len(a))
    b += (0,) * (n - len(b))
    return a > b


def _opener():
    ctx = ssl.create_default_context()
    try:
        import certifi  # neu co
        ctx.load_verify_locations(certifi.where())
    except Exception:  # noqa
        pass
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def check_latest(repo, timeout=15):
    """Tra ve dict {version, name, notes, asset_url, asset_name, page} cua ban moi nhat."""
    req = urllib.request.Request(API.format(repo=repo),
                                 headers={"User-Agent": "NR-Viewer-Updater",
                                          "Accept": "application/vnd.github+json"})
    with _opener().open(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    assets = data.get("assets", [])
    # uu tien file .exe, roi .zip, roi asset dau tien
    pick = None
    for want in (".exe", ".zip"):
        for a in assets:
            if a.get("name", "").lower().endswith(want):
                pick = a
                break
        if pick:
            break
    if pick is None and assets:
        pick = assets[0]
    return {
        "version": data.get("tag_name", "") or data.get("name", ""),
        "name": data.get("name", ""),
        "notes": data.get("body", "") or "",
        "asset_url": pick.get("browser_download_url") if pick else None,
        "asset_name": pick.get("name") if pick else None,
        "asset_size": pick.get("size", 0) if pick else 0,
        "page": data.get("html_url", ""),
    }


def download(url, dest, cb=None, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "NR-Viewer-Updater"})
    with _opener().open(req, timeout=timeout) as r:
        total = int(r.headers.get("Content-Length", 0) or 0)
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if cb:
                    cb(done, total)
    return dest


def is_frozen():
    return getattr(sys, "frozen", False)


def current_exe():
    return os.path.abspath(sys.executable)


def apply_update_and_restart(new_file):
    """
    Thay file .exe dang chay bang new_file roi khoi dong lai (chi Windows, ban --onefile).
    Viet 1 file .bat cho: doi app thoat -> ghi de -> mo lai.
    """
    if os.name != "nt" or not is_frozen():
        raise RuntimeError("Chi tu cap nhat duoc voi ban .exe tren Windows")
    exe = current_exe()
    folder = os.path.dirname(exe)
    bat = os.path.join(folder, "_nr_update.bat")
    # doi process cu thoat (move se that bai khi file con khoa) roi thu lai
    content = (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        ":retry\r\n"
        f'move /y "{new_file}" "{exe}" >nul 2>&1\r\n'
        "if errorlevel 1 (\r\n"
        "  ping 127.0.0.1 -n 2 >nul\r\n"
        "  goto retry\r\n"
        ")\r\n"
        f'start "" "{exe}"\r\n'
        'del "%~f0"\r\n'
    )
    with open(bat, "w", encoding="utf-8") as f:
        f.write(content)
    flags = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS
    subprocess.Popen(["cmd", "/c", bat], creationflags=flags, close_fds=True)
    return True
