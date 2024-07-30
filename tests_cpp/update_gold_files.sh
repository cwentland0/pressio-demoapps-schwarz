#!/bin/bash

run=0

root_dir="/home/crwentl/research/code/pressio-proj/pressio-demoapps-schwarz"
build_dir="build"
test_dir="tests_cpp"

ndoms=4
ndoms_parallel=12


# standard Schwarz tests
declare -a schwarz_tests=(
    "eigen_2d_euler_riemann_implicit_schwarz"
    "eigen_2d_swe_slip_wall_implicit_schwarz"
    "eigen_2d_swe_slip_wall_implicit_roms_schwarz/lspg"
    "eigen_2d_swe_slip_wall_implicit_hproms_schwarz/lspg"
    "eigen_2d_swe_slip_wall_implicit_mixed_schwarz/lspg"
    "eigen_2d_swe_slip_wall_implicit_nonoverlap_dd"
    "eigen_2d_swe_slip_wall_implicit_hproms_gpod_schwarz/lspg"
    "eigen_2d_swe_slip_wall_implicit_roms_schwarz_icFile/lspg"
)
declare -a schwarz_orders=(
    "firstorder"
    "weno3"
)

# parallel Schwarz tests
declare -a schwarz_parallel_tests=(
    "eigen_2d_swe_slip_wall_implicit_schwarz_parallel"
    "eigen_2d_swe_slip_wall_implicit_hproms_schwarz_parallel/lspg"
)
declare -a schwarz_parallel_orders=(
    "firstorder"
    "weno3"
)

# --------------

declare -a goldroots

# Schwarz tests
for case_dir in "${schwarz_tests[@]}"; do
    if [[ "${case_dir}" == *"euler"* ]]; then
        goldroots=("p" "rho")
    elif [[ "${case_dir}" == *"swe"* ]]; then
        goldroots=("h")
    else
        echo "Invalid case_dir: ${case_dir}"
        exit
    fi

    for order in "${schwarz_orders[@]}"; do
        gold_dir="${root_dir}/${build_dir}/${test_dir}/${case_dir}/${order}"
        targ_dir="${root_dir}/${test_dir}/${case_dir}/${order}"
        for ((dom=0;dom<${ndoms};++dom)); do
            for gold_root in "${goldroots[@]}"; do
                gold_file="${gold_root}_${dom}.txt"
                targ_file="${gold_root}_gold_${dom}.txt"
                if [[ ${run} == 1 ]]; then
                    mv -v "${gold_dir}/${gold_file}" "${targ_dir}/${targ_file}"
                else
                    ls "${gold_dir}/${gold_file}"
                    ls "${targ_dir}/${targ_file}"
                fi
            done
        done
    done
done

# parallel Schwarz tests
for case_dir in "${schwarz_parallel_tests[@]}"; do
    if [[ "${case_dir}" == *"euler"* ]]; then
        goldroots=("p" "rho")
    elif [[ "${case_dir}" == *"swe"* ]]; then
        goldroots=("h")
    else
        echo "Invalid case_dir: ${case_dir}"
        exit
    fi

    for order in "${schwarz_parallel_orders[@]}"; do
        # just assume OMP is correct
        gold_dir="${root_dir}/${build_dir}/${test_dir}/parallel/${case_dir}/${order}_omp"
        targ_dir="${root_dir}/${test_dir}/parallel/${case_dir}/${order}"
        for ((dom=0;dom<${ndoms_parallel};++dom)); do
            for gold_root in "${goldroots[@]}"; do
                gold_file="${gold_root}_${dom}.txt"
                targ_file="${gold_root}_gold_${dom}.txt"
                if [[ ${run} == 1 ]]; then
                    mv -v "${gold_dir}/${gold_file}" "${targ_dir}/${targ_file}"
                else
                    ls "${gold_dir}/${gold_file}"
                    ls "${targ_dir}/${targ_file}"
                fi
            done
        done
    done
done
