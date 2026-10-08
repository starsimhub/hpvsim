# HPVsim installers

Standalone installers for users who don't already have Python set up. They install
Miniconda (if conda isn't already present) and then `pip install hpvsim jupyter`
into the base conda environment.

If you already manage your own Python environment, you don't need these — just
`pip install hpvsim` as described in the main README.

## Linux / macOS

One-liner (recommended):

```bash
curl -fsSL https://raw.githubusercontent.com/starsimhub/hpvsim/main/install/install.sh | bash
```

Or download and run `install.sh`:

```bash
bash install.sh
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

1. Check for an existing `conda` on PATH or in the default Miniconda location
   (`~/miniconda3` on Unix, `%USERPROFILE%\Miniconda3` on Windows). If found,
   use it.
2. Otherwise download the latest Miniconda installer for your OS/architecture
   from `repo.anaconda.com` and install silently into the default location.
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
