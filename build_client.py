"""
FortifAI Client Build Script
Packages client_agent.py into standalone executables
"""

import PyInstaller.__main__
import sys
import os

def build_client():
    """Build client agent as standalone executable"""
    
    # Determine platform
    platform = sys.platform
    
    # Common options
    common_options = [
        'client_agentAI.py',
        '--name=FortifAI_Client',
        '--onedir',  # Single executable
        '--windowed',  # No console (remove for debugging)
        '--icon=icon.ico',  # Add your icon
        
        # Hidden imports (CRITICAL for TensorFlow/sklearn)
        '--hidden-import=tensorflow',
        '--hidden-import=sklearn',
        '--hidden-import=sklearn.ensemble',
        '--hidden-import=sklearn.svm',
        '--hidden-import=sklearn.preprocessing',
        '--hidden-import=scipy',
        '--hidden-import=scipy.stats',
        '--hidden-import=numpy',
        '--hidden-import=psutil',
        '--hidden-import=tkinter',
        '--hidden-import=pickle',
        '--hidden-import=socket',
        '--hidden-import=threading',
        '--hidden-import=queue',
        '--hidden-import=collections',
        '--hidden-import=datetime',
        '--hidden-import=platform',
        '--hidden-import=hashlib',
        '--hidden-import=json',
        '--hidden-import=codecs',
        
        # Collect data files
        '--collect-data=tensorflow',
        '--collect-data=sklearn',
        '--collect-all=scipy',
        
        # Exclude unnecessary modules
        '--exclude-module=matplotlib',
        '--exclude-module=pandas',
        '--exclude-module=jupyter',
        '--exclude-module=notebook',
        '--exclude-module=IPython',
    ]
    
    # Platform-specific options
    if platform == 'win32':
        print("Building Windows .exe...")
        PyInstaller.__main__.run(common_options + [
            '--add-data=README.md;.',  # Windows uses ;
        ])
    else:
        print("Building Linux binary...")
        PyInstaller.__main__.run(common_options + [
            '--add-data=README.md:.',  # Linux uses :
        ])
    
    print("\n" + "="*60)
    print("✓ Build complete!")
    print(f"Executable location: dist/FortifAI_Client{'exe' if platform == 'win32' else ''}")
    print("="*60)

if __name__ == '__main__':
    build_client()