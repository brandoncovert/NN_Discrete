#!/bin/bash

# Clean up old environment
rm -rf .venv

# Detect the operating system
OS="$(uname)"

if [ "$OS" = "Darwin" ]; then
    # ==========================================
    # LOCAL MAC SETUP (CPU + OpenMP Fix)
    # ==========================================
    echo "macOS detected: Installing CPU version and fixing OpenMP..."
    
    uv sync --extra cpu --reinstall

    # Delete redundant Mac libraries
    rm .venv/lib/python3.12/site-packages/quspin_extensions/.dylibs/libomp.dylib
    rm .venv/lib/python3.12/site-packages/parallel_sparse_tools/.dylibs/libomp.dylib

    # Create symlinks to Homebrew OpenMP
    ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib .venv/lib/python3.12/site-packages/quspin_extensions/.dylibs/libomp.dylib
    ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib .venv/lib/python3.12/site-packages/parallel_sparse_tools/.dylibs/libomp.dylib

elif [ "$OS" = "Linux" ]; then
    # ==========================================
    # HIPERGATOR SETUP (GPU)
    # ==========================================
    echo "Linux detected: Installing GPU version..."
    
    # HiPerGator shouldn't need the OpenMP symlink hack
    uv sync --extra gpu --reinstall

else
    echo "Unknown OS: $OS. Exiting."
    exit 1
fi

echo "Environment setup complete!"