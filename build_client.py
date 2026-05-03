"""
FortifAI Client Build Script
Packages client_agentAI.py into a standalone application (PyInstaller onedir).
Run from the repository root: python build_client.py
"""

import os
import sys

import PyInstaller.__main__


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def build_client() -> None:
    """Build the FortifAI client agent as a GUI application bundle."""
    root = _repo_root()
    os.chdir(root)

    platform_name = sys.platform
    is_win = platform_name == "win32"

    sep = ";" if is_win else ":"
    extra_data: list[str] = []
    readme = os.path.join(root, "README.md")
    if os.path.isfile(readme):
        extra_data.append(f"--add-data={readme}{sep}.")
    env_file = os.path.join(root, ".env")
    if os.path.isfile(env_file):
        extra_data.append(f"--add-data={env_file}{sep}.")

    icon_opt: list[str] = []
    for icon_name in ("icon.ico", os.path.join("client", "icon.ico")):
        icon_path = os.path.join(root, icon_name)
        if os.path.isfile(icon_path):
            icon_opt = [f"--icon={icon_path}"]
            break

    # Paths so `client.*` and `shared.*` resolve during analysis
    path_opts = [f"--paths={root}"]

    common_options = [
        os.path.join("client", "client_agentAI.py"),
        "--name=FortifAI_Client",
        "--onedir",
        "--windowed",
        "--noconfirm",
        *icon_opt,
        # PyQt5 GUI + charts (client/gui/client_gui.py)
        "--hidden-import=PyQt5",
        "--hidden-import=PyQt5.QtCore",
        "--hidden-import=PyQt5.QtGui",
        "--hidden-import=PyQt5.QtWidgets",
        "--hidden-import=PyQt5.QtChart",
        "--collect-all=PyQt5",
        "--collect-all=PyQtChart",
        # Gemini + env
        "--hidden-import=google.generativeai",
        "--hidden-import=google.api_core",
        "--hidden-import=dotenv",
        # Shared package (imported via client / warning_filters)
        "--hidden-import=shared",
        "--hidden-import=shared.constants",
        "--hidden-import=shared.protocol",
        "--hidden-import=shared.detection_config",
        "--hidden-import=shared.warning_filters",
        # ML stack
        "--hidden-import=tensorflow",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.ensemble",
        "--hidden-import=sklearn.svm",
        "--hidden-import=sklearn.preprocessing",
        "--hidden-import=scipy",
        "--hidden-import=scipy.stats",
        "--hidden-import=numpy",
        "--hidden-import=psutil",
        "--hidden-import=pickle",
        "--hidden-import=socket",
        "--hidden-import=threading",
        "--hidden-import=queue",
        "--hidden-import=collections",
        "--hidden-import=datetime",
        "--hidden-import=platform",
        "--hidden-import=hashlib",
        "--hidden-import=json",
        "--hidden-import=codecs",
        "--collect-data=tensorflow",
        "--collect-data=sklearn",
        "--collect-all=scipy",
        *path_opts,
        "--exclude-module=matplotlib",
        "--exclude-module=pandas",
        "--exclude-module=jupyter",
        "--exclude-module=notebook",
        "--exclude-module=IPython",
        *extra_data,
    ]

    if is_win:
        print("Building Windows application (onedir)...")
    else:
        print("Building Linux application (onedir)...")

    PyInstaller.__main__.run(common_options)

    exe_suffix = os.path.join("FortifAI_Client.exe") if is_win else "FortifAI_Client"
    out = os.path.join(root, "dist", "FortifAI_Client", exe_suffix)
    print("\n" + "=" * 60)
    print("Build complete.")
    print(f"Application: {out}")
    print("=" * 60)


if __name__ == "__main__":
    build_client()
