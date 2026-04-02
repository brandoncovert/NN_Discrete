rm -rf .venv

# 1. Reinstall without the broken binaries
uv sync --reinstall

# 2. Delete the redundant libraries uv just downloaded
rm .venv/lib/python3.12/site-packages/quspin_extensions/.dylibs/libomp.dylib
rm .venv/lib/python3.12/site-packages/parallel_sparse_tools/.dylibs/libomp.dylib

# 3. Create the symlinks to your Homebrew OpenMP
ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib .venv/lib/python3.12/site-packages/quspin_extensions/.dylibs/libomp.dylib
ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib .venv/lib/python3.12/site-packages/parallel_sparse_tools/.dylibs/libomp.dylib