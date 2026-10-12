# HPVsim installers

Standalone installers for users who don't already have Python set up. They install
Miniforge (if conda isn't already present) and then `pip install hpvsim jupyter`
into the base conda environment.

If you already manage your own Python environment, you don't need these — just
`pip install hpvsim` as described in the main README.

## Linux / macOS

One-liner (recommended):

```bash
curl -LsSf https://hpvsim.org/install.sh | sh
```

Or download and run `install.sh`:

```bash
sh install.sh
```

Open a new terminal afterwards so `conda` is on your PATH.

## Windows

Download [`install.bat`](install.bat) and double-click it, or run from Command
Prompt:

```cmd
install.bat
```

Open a new Command Prompt afterwards so `conda` is on your PATH.

## What the installers do

1. Check for an existing `conda` on PATH or in the default Miniforge location
   (`~/miniforge3` on Unix, `%USERPROFILE%\miniforge3` on Windows). If found,
   use it.
2. Otherwise download the latest Miniforge installer for your OS/architecture
   from the [conda-forge/miniforge](https://github.com/conda-forge/miniforge)
   GitHub releases and install silently into the default location.
   On Unix, `conda init` is run so future shells see conda; on Windows, the
   installer is called with `/AddToPath=1`.
3. `pip install --upgrade pip` and `pip install hpvsim jupyter` into the base
   conda env.

## Verifying

Open a new terminal and run:

```
python -c "import hpvsim; print(hpvsim.__version__)"
jupyter notebook
```
