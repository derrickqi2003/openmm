#!/bin/bash
# Build this branch of OpenMM with CUDA on a Linux NVIDIA machine, run OpenMM's CUDA tests with and without the
# pruned neighbor list, then the correctness checks and the speed benchmark on the peptoid box.
#
# usage: bash perf/engine/build_and_test.sh REPO [WORK]
#   REPO  a for-derrick-q checkout (needs build/ and npt/; npt/state_post_npt_equil.xml is used if present)
#   WORK  scratch directory for the environment and build (default: /workspace/engine-work)
# Run it from the root of this OpenMM checkout.  It installs micromamba, the compilers' helpers and, if the machine
# has no CUDA toolkit, one from conda-forge that is no newer than the driver supports.  About 15 minutes.
# With SHORT=1 it runs only the tests most affected by the changes, a shorter correctness check, and the benchmark of
# the baseline and each change added in turn.
set -euo pipefail
SRC=$(pwd)
REPO=$(cd "${1:?usage: build_and_test.sh REPO [WORK]}" && pwd)
WORK=${2:-/workspace/engine-work}
mkdir -p "$WORK"
ENV=$WORK/env
if ! command -v micromamba >/dev/null; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$WORK" bin/micromamba
    export PATH=$WORK/bin:$PATH
fi

# The CUDA toolkit must not be newer than the driver, or kernels fail to load (CUDA_ERROR_UNSUPPORTED_PTX_VERSION).
DRIVER_CUDA=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | awk '{print $3}')
echo "driver supports CUDA $DRIVER_CUDA"
PKGS="python=3.11 cmake ninja swig doxygen numpy cython setuptools pip"
if [ ! -d "$ENV" ]; then
    micromamba create -y -p "$ENV" -c conda-forge $PKGS "cuda-version<=$DRIVER_CUDA" cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev \
        cuda-driver-dev cuda-profiler-api libcufft-dev
fi
export PATH=$ENV/bin:$PATH
CUDA_ROOT=$ENV
if [ -d "$ENV/targets/x86_64-linux" ]; then
    CUDA_INC=$ENV/targets/x86_64-linux/include
else
    CUDA_INC=$ENV/include
fi

BUILD=$WORK/build; PREFIX=$WORK/install
mkdir -p "$BUILD"
cd "$BUILD"
if [ ! -f build.ninja ]; then
    cmake -G Ninja "$SRC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DOPENMM_BUILD_CUDA_LIB=ON -DOPENMM_BUILD_CUDA_TESTS=ON -DOPENMM_BUILD_OPENCL_LIB=OFF -DOPENMM_BUILD_HIP_LIB=OFF \
        -DOPENMM_BUILD_AMOEBA_CUDA_LIB=OFF -DOPENMM_BUILD_RPMD_CUDA_LIB=OFF -DOPENMM_BUILD_DRUDE_CUDA_LIB=OFF \
        -DOPENMM_BUILD_EXAMPLES=OFF -DOPENMM_BUILD_C_AND_FORTRAN_WRAPPERS=OFF -DBUILD_TESTING=ON \
        -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" -DCMAKE_CUDA_COMPILER="$ENV/bin/nvcc" \
        -DPYTHON_EXECUTABLE="$ENV/bin/python" -DSWIG_EXECUTABLE="$ENV/bin/swig" -DDOXYGEN_EXECUTABLE="$ENV/bin/doxygen"
fi
ninja -j "$(nproc)" install
OPENMM_INCLUDE_PATH=$PREFIX/include OPENMM_LIB_PATH=$PREFIX/lib ninja PythonInstall
export OPENMM_PLUGIN_DIR=$PREFIX/lib/plugins
export LD_LIBRARY_PATH=$PREFIX/lib:$ENV/lib:${LD_LIBRARY_PATH:-}
python -m openmm.testInstallation

echo; echo "=== OpenMM CUDA tests, ordinary and pruned neighbor list"
if [ "${SHORT:-0}" = 1 ]; then
TESTS="TestCudaNonbondedForce TestCudaCustomNonbondedForce TestCudaEwald TestCudaLangevinMiddleIntegrator TestCudaMonteCarloBarostat"
else
TESTS="TestCudaNonbondedForce TestCudaCustomNonbondedForce TestCudaEwald TestCudaDispersionPME TestCudaGBSAOBCForce
       TestCudaCustomGBForce TestCudaCustomHbondForce TestCudaConstantPotentialForce TestCudaLCPOForce TestCudaMultipleForces
       TestCudaLangevinMiddleIntegrator TestCudaMonteCarloBarostat TestCudaMonteCarloFlexibleBarostat TestCudaCheckpoints"
fi
FAILED=""
for t in $TESTS; do
    [ -x "$BUILD/$t" ] || continue
    for prune in 0 1; do
        if OPENMM_CUDA_PRUNED_NEIGHBOR_LIST=$prune "$BUILD/$t" > "$WORK/$t.prune$prune.log" 2>&1; then r=ok; else r=FAILED; FAILED="$FAILED $t(prune=$prune)"; fi
        printf "%-36s pruning=%s  %s\n" "$t" "$prune" "$r"
    done
done

STATE=$REPO/npt/state_post_npt_equil.xml
STATEARG=""
[ -f "$STATE" ] && STATEARG="--state $STATE"
echo; echo "=== Correctness on the peptoid box"
if [ "${SHORT:-0}" = 1 ]; then
    CHECKARGS=(--steps 1000)
    BENCHARGS=(--main --steps 4000 --repeats 5)
else
    CHECKARGS=()
    BENCHARGS=(--steps 4000 --repeats 5 --only "baseline;defaults;without split build;without padding 0.12 nm;without reorder every 1000;without build prefilter;without list growth;without warp block bounds;without PME sort every 20;reciprocal every 2 steps;pruned, anchored list;baseline;defaults")
fi
python "$SRC/perf/engine/engine_check.py" --repo "$REPO" $STATEARG ${CHECKARGS[@]+"${CHECKARGS[@]}"} || FAILED="$FAILED engine_check"
echo; echo "=== Speed on the peptoid box (each variant in its own process)"
python "$SRC/perf/engine/engine_bench.py" --repo "$REPO" $STATEARG "${BENCHARGS[@]}"
echo
if [ -n "$FAILED" ]; then echo "FAILURES:$FAILED (logs in $WORK)"; exit 1; fi
echo "everything passed"
