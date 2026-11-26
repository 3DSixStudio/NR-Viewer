NR Viewer - xem truoc file .nr (Ninja Ripper 2.x) tren Windows
=================================================================

Cai dat / chay
  1. Can Python 3.10+ (tick "Add to PATH" khi cai).
  2. Double-click run.bat  (lan dau se tu cai PySide6, PyOpenGL, numpy).
     Hoac: run.bat "F:\NinjaRipper\...\frame_0"  de mo thang thu muc.
  3. Muon co file .exe: chay build_exe.bat -> dist\NRViewer\NRViewer.exe

Su dung
  - Mo thu muc frame_N (hoac thu muc cha chua frame_N).
  - Danh sach ben trai: click 1 mesh de xem, Ctrl/Shift de chon nhieu.
    Bam vao tieu de cot de sap xep theo so tam giac, so dinh...
  - "An quad / UI 2D": an cac mesh UI/full-screen.
  - Che do "Mesh goc": hinh hoc object space (dau vao vertex shader).
    "Canh dung lai": dung lai vi tri tu SV_Position -> thay ca canh nhu trong game.
    Nut "Dung ca frame" nap tat ca mesh dang loc. "Camera goc" = goc nhin cua game.
  - Panel Texture & UV: doi texture/kenh UV neu tool doan sai
    (mac dinh lay texture it dung chung nhat = thuong la albedo).
  - Xuat FBX (kem material + texture) de mang vao Maya / Blender / Unity.
    Texture chep vao thu muc <ten>_textures canh file FBX, hoac nhung luon vao FBX.

Chuot: trai = xoay, giua hoac Shift+trai = pan, lan/phai = zoom, F = frame.

Tu dong cap nhat: xem CAP_NHAT_GITHUB.txt (nut "Kiem tra cap nhat" goc tren phai).

Dinh dang .nr (da phan tich): xem docstring dau file nr_format.py.
