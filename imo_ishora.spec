# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec fayli — Imo-Ishora Tarjimoni EXE (PyInstaller 6.x)
"""

import os, sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_all

BASE = os.path.dirname(os.path.abspath(SPEC))

# ── Ma'lumot fayllar ──────────────────────────────────────────────────────────
datas = [
    (os.path.join(BASE, 'models'),       'models'),
    (os.path.join(BASE, 'data'),         'data'),
    (os.path.join(BASE, 'audio_cache'),  'audio_cache'),
    (os.path.join(BASE, 'src'),          'src'),
    (os.path.join(BASE, 'settings.json'), '.'),
    (os.path.join(BASE, 'app.py'),           '.'),
    (os.path.join(BASE, 'gui_collect.py'),   '.'),
    (os.path.join(BASE, 'gui_train.py'),     '.'),
    (os.path.join(BASE, 'gui_settings.py'),  '.'),
]

# customtkinter assets
try:
    import customtkinter
    ctk_path = os.path.dirname(customtkinter.__file__)
    datas += [(ctk_path, 'customtkinter')]
except Exception:
    pass

# mediapipe — collect_all ensures DLLs are on real filesystem for importlib.resources
mp_datas, mp_binaries, mp_hidden = collect_all('mediapipe')
datas    += mp_datas


# matplotlib fonts/styles
datas += collect_data_files('matplotlib')

# ── Hidden imports ────────────────────────────────────────────────────────────
hiddenimports = [
    'customtkinter',
    'customtkinter.windows',
    'customtkinter.windows.widgets',
    'customtkinter.windows.widgets.core_rendering',
    'darkdetect',
    'mediapipe',
    'mediapipe.tasks',
    'mediapipe.tasks.python',
    'mediapipe.tasks.python.vision',
    'mediapipe.tasks.python.core',
    'mediapipe.tasks.python.core.base_options',
    'mediapipe.tasks.core',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.ensemble._forest',
    'sklearn.ensemble._gb',
    'sklearn.svm',
    'sklearn.pipeline',
    'sklearn.preprocessing',
    'sklearn.model_selection',
    'sklearn.metrics',
    'sklearn.utils._cython_blas',
    'sklearn.utils._weight_vector',
    'sklearn.neighbors._typedefs',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils',
    'cv2',
    'pygame',
    'pygame.mixer',
    'gtts',
    'pandas',
    'numpy',
    'joblib',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'matplotlib',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure',
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.simpledialog',
    'tkinter.filedialog',
    'threading',
    'collections',
    'json',
    'csv',
]

# ── Binaries ──────────────────────────────────────────────────────────────────
binaries = []
binaries += mp_binaries

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    [os.path.join(BASE, 'launcher.py')],
    pathex=[BASE],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + mp_hidden,
    hookspath=[os.path.join(BASE, 'hooks')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython', 'jupyter', 'notebook',
        'scipy', 'sympy',
        'tensorflow', 'tensorflow_core',
        'tensorboard', 'tf2onnx',
        'jax', 'flax', 'torch', 'torchvision',
    ],
    noarchive=True,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ImoIshora',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ImoIshora',
)
