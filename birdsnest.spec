# -*- mode: python ; coding: utf-8 -*-
"""
Bird's Nest — PyInstaller spec file.

Build with: pyinstaller birdsnest.spec
Output: dist/BirdsNest.app
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── Paths ──
PROJ_DIR = os.path.abspath('.')

# ── Data files to bundle ──
datas = [
    # Web UI
    ('birdsnest/web', 'birdsnest/web'),
    # Version file
    ('VERSION', '.'),
]

# ── Hidden imports PyInstaller misses ──
hiddenimports = [
    # Server
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'fastapi',
    'starlette',
    'starlette.routing',
    'starlette.middleware',
    'starlette.middleware.cors',
    'starlette.responses',
    'starlette.staticfiles',
    'starlette.websockets',
    'anyio',
    'anyio._backends',
    'anyio._backends._asyncio',
    'httptools',
    'websockets',

    # ML
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'numpy',
    'tokenizers',
    'psutil',

    # Birdsnest package
    'birdsnest',
    'birdsnest.server',
    'birdsnest.engine',
    'birdsnest.engines',
    'birdsnest.engines.rwkv_engine',
    'birdsnest.engines.hf_engine',
    'birdsnest.models',
    'birdsnest.tools',
    'birdsnest.rag',
]

# ── Exclude heavy stuff we don't need ──
excludes = [
    'mamba_ssm',           # Not installed on most systems
    'mflux',               # Optional image gen — install separately
    'matplotlib',
    'scipy',
    'pandas',
    'PIL',
    'cv2',
    'tkinter',
    'PyQt5',
    'PyQt6',
    'PySide2',
    'PySide6',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'setuptools',
    'pip',
]


a = Analysis(
    ['launcher.py'],
    pathex=[PROJ_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BirdsNest',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX doesn't help much on macOS
    console=False,  # No terminal window
    target_arch='arm64',  # Apple Silicon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='BirdsNest',
)

app = BUNDLE(
    coll,
    name="Bird's Nest.app",
    icon='assets/icon.icns',
    bundle_identifier='com.birdsnest.app',
    info_plist={
        'CFBundleName': "Bird's Nest",
        'CFBundleDisplayName': "Bird's Nest",
        'CFBundleShortVersionString': open('VERSION').read().strip(),
        'CFBundleVersion': open('VERSION').read().strip(),
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '13.0',  # macOS Ventura+ (MPS requirement)
        'NSAppleEventsUsageDescription': 'Bird\'s Nest needs to open your browser.',
    },
)
