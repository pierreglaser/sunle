#!/usr/bin/env bash
mkdir -p .env

# Setup a conda-based package manager (mambaforge in that case) locally
curl -L  -o "./.env/Mambaforge-$(uname)-$(uname -m).sh" "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash ".env/Mambaforge-$(uname)-$(uname -m).sh" -b -p ./.env/mambaforge

# Enabble conda within this shell
eval "$('.env/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# Install all dependencies to run the experiments within a conda virtual environment named $env_name
# Look for a --gpu flag. If it's there, use the GPU environment (environment-gpu.yml), otherwise use the CPU-only environment (environment-cpu.yml)
if [[ " $* " =~ " --gpu " ]]; then
    echo "Using GPU environment"
    env_name="unle-gpu"
    .env/mambaforge/bin/mamba env create -f environment-gpu.yml
else
    echo "Using CPU environment"
    env_name="unle-cpu"
    .env/mambaforge/bin/mamba env create -f environment-cpu.yml
fi

# Pass some julia environment variables to the environment
mkdir -p .env/mambaforge/envs/${env_name}/etc/conda/activate.d
mkdir -p .env/mambaforge/envs/${env_name}/etc/conda/deactivate.d
printf "export JULIA_SYSIMAGE_DIFFEQTORCH=$(pwd)/.env/.julia_sysimage_diffeqtorch.so" > .env/mambaforge/envs/${env_name}/etc/conda/activate.d/env_vars.sh
printf "unset JULIA_SYSIMAGE_DIFFEQTORCH" > .env/mambaforge/envs/${env_name}/etc/conda/deactivate.d/env_vars.sh

# Activate the virtual environment
conda activate ${env_name}

# Install vendored libraries
python -m pip install -e ../src/unle/

python -m pip install -e ./src/density-utils/
python -m pip install -e ./src/experiments_utils/
python -m pip install -e ./src/jax-utils/
python -m pip install -e ./src/pck3/
python -m pip install -e ./src/pyloric/
python -m pip install -e ./src/sbibm/
python -m pip install -e ./src/smnle/
python -m pip install -e ./src/sbibm-unle-extra/


# Install the diffeqtorch package (used to simulate from the `lotka_volterra` model)
# This takes approximately 15 minutes to run
python -c "from diffeqtorch.install import install_and_test; install_and_test()"
