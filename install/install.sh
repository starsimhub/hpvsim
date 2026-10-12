#!/usr/bin/env bash
# HPVsim installer for Linux and macOS.
# Installs Miniforge (if needed) and pip-installs hpvsim + jupyter.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/starsimhub/hpvsim/main/install/install.sh | bash
# or:
#   bash install.sh

set -euo pipefail

MINIFORGE_DIR="${HOME}/miniforge3"

detect_installer() {
    local os arch
    case "$(uname -s)" in
        Linux)  os="Linux" ;;
        Darwin) os="MacOSX" ;;
        *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
    esac
    case "$(uname -m)" in
        x86_64)
            arch="x86_64" ;;
        arm64|aarch64)
            if [ "$os" = "Linux" ]; then arch="aarch64"; else arch="arm64"; fi ;;
        *)
            echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    echo "Miniforge3-${os}-${arch}.sh"
}

find_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
    elif [ -x "${MINIFORGE_DIR}/bin/conda" ]; then
        echo "${MINIFORGE_DIR}/bin/conda"
    fi
}

install_miniforge() {
    local installer url tmpdir
    installer="$(detect_installer)"
    url="https://github.com/conda-forge/miniforge/releases/latest/download/${installer}"
    tmpdir="$(mktemp -d)"  # Miniforge's installer requires its filename to end in .sh

    echo ">>> Downloading ${url}"
    curl -fsSL "${url}" -o "${tmpdir}/${installer}"

    echo ">>> Installing Miniforge to ${MINIFORGE_DIR}"
    bash "${tmpdir}/${installer}" -b -p "${MINIFORGE_DIR}"
    rm -rf "${tmpdir}"

    echo ">>> Initializing conda for bash and zsh"
    "${MINIFORGE_DIR}/bin/conda" init bash zsh >/dev/null || true
}

main() {
    echo "=== HPVsim installer ==="

    local conda
    conda="$(find_conda || true)"
    if [ -z "${conda}" ]; then
        install_miniforge
        conda="${MINIFORGE_DIR}/bin/conda"
    else
        echo ">>> Found existing conda at: ${conda}"
    fi

    echo ">>> Installing hpvsim and jupyter into the base conda env"
    "${conda}" run --no-capture-output -n base pip install --upgrade pip
    "${conda}" run --no-capture-output -n base pip install hpvsim jupyter

    cat <<EOF

=== Done ===
Open a new terminal and verify with:
    python -c "import hpvsim; print(hpvsim.__version__)"

Launch Jupyter with:
    jupyter notebook
EOF
}

main "$@"
