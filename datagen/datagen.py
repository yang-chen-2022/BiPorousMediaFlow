"""Generate AMITEX inputs, run simulations, and collate permeability results."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .fft_ma import fft_ma_2d
from .helper import postProcBrinkman_readVFIELD


RESOLUTION = 64
OUTPUT_DIR = Path(
    os.environ.get(
        "DATAGEN_OUTPUT_DIR",
        f"/mnt/data/ML/StokesBrinkman/R{RESOLUTION}_unseen",
    )
)
AMITEX_ENV = os.environ.get(
    "AMITEX_ENV", "/home/projects/AMITEX/amitex_fftp/env_amitex.sh"
)
AMITEX_EXECUTABLE = os.environ.get(
    "AMITEX_EXECUTABLE",
    "/home/projects/AMITEX/amitex_brinkman/BRINKMAN1/src/amitex_fftp",
)
MATERIAL_LIBRARY = os.environ.get(
    "AMITEX_MATERIAL_LIBRARY",
    "/home/projects/AMITEX/amitex_brinkman/BRINKMAN1/src/behavior/libUmatAmitex.so",
)

MEAN_VALUES = np.linspace(-11, -4, 8)
STANDARD_DEVIATIONS = np.logspace(0, 0, 1)  # This will create an array with a single value 1.0
SCALE_X_VALUES = np.linspace(0.05, 0.5, 5)
SCALE_Y_VALUES = np.linspace(0.05, 0.5, 5)
ANGLES = np.linspace(0, 135, 4)
REPEATS = 1
FLUID_THRESHOLD_QUANTILE = 0.8
CPU_COUNT = 6


def _sample_suffix(mean_value, stdev, scale_x, scale_y, angle, repeat):
    return (
        f"_kM{10**mean_value:.3E}"
        f"_kS{stdev:.3E}"
        f"_scx{scale_x:.3E}"
        f"_scy{scale_y:.3E}"
        f"_ang{angle:.3E}"
        f"_UC{repeat}"
    )


def generate_inputs(output_dir=OUTPUT_DIR):
    """Create meshes, material files, and launch scripts for the parameter sweep."""
    from .IOfcts import (
        saveMesh2VTK_amitex,
        write_AMITEX_launcher_ml,
        write_AMITEX_xml_algo,
        write_AMITEX_xml_load,
        write_AMITEX_xml_mat,
    )

    output_dir = Path(output_dir)
    for directory in ("mesh", "zoneID", "mate", "script", "resu"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    length = 1.0
    nx = ny = RESOLUTION
    dx = dy = length / RESOLUTION
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dy, dx)
    viscosity = 1.0
    effective_viscosity = 1.0
    algorithm_parameters = np.array(
        [
            1,       # Finite difference scheme
            1,       # Velocity control
            1e-7,    # Convergence criterion
            1,       # Activate convergence acceleration
            2,       # Apply acceleration every two iterations
            100000,  # Maximum number of iterations
            viscosity,
            effective_viscosity,
            np.inf,
            0,
            0,
        ]
    )

    algorithm_file = output_dir / "algo_generic.xml"
    load_file = output_dir / "load_generic.xml"
    write_AMITEX_xml_algo(str(algorithm_file), algorithm_parameters)
    write_AMITEX_xml_load(str(load_file))

    for mean_value in MEAN_VALUES:
        for stdev in STANDARD_DEVIATIONS:
            for scale_x in SCALE_X_VALUES:
                for scale_y in SCALE_Y_VALUES:
                    for angle in ANGLES:
                        for repeat in range(REPEATS):
                            suffix = _sample_suffix(
                                mean_value, stdev, scale_x, scale_y, angle, repeat
                            )
                            random_field = fft_ma_2d(
                                nx=nx,
                                ny=ny,
                                dx=dx,
                                dy=dy,
                                mean_value=mean_value,
                                stdev=stdev,
                                scale=[scale_x, scale_y],
                                angle=angle,
                            )
                            permeability = 10**random_field
                            threshold = np.quantile(
                                permeability, FLUID_THRESHOLD_QUANTILE
                            )
                            is_fluid = permeability > threshold
                            beta = viscosity / permeability
                            beta[is_fluid] = np.nan

                            coefficients = [
                                [
                                    effective_viscosity,
                                    beta[~is_fluid],
                                    beta[~is_fluid],
                                    beta[~is_fluid],
                                    0,
                                    0,
                                    0,
                                ],
                                [viscosity, 0, 0, 0, 0, 0, 0],
                            ]
                            reference_beta = 0.5 * beta[~is_fluid].max()

                            mesh_file = output_dir / "mesh" / f"mID{suffix}.vtk"
                            mesh = np.ones(beta.shape, dtype=np.uint8)
                            mesh[is_fluid] = 2
                            saveMesh2VTK_amitex(
                                str(mesh_file), mesh, "matID",
                                origin=origin, spacing=spacing
                            )

                            zone_file = output_dir / "zoneID" / f"zID{suffix}.vtk"
                            zone_ids = np.ones(beta.shape, dtype=np.uint32)
                            solid_zone_count = np.count_nonzero(~is_fluid)
                            zone_ids[~is_fluid] = np.arange(solid_zone_count) + 1
                            saveMesh2VTK_amitex(
                                str(zone_file), zone_ids, "zoneID",
                                origin=origin, spacing=spacing
                            )

                            material_file = output_dir / "mate" / f"mate{suffix}.xml"
                            write_AMITEX_xml_mat(
                                str(material_file),
                                reference_beta,
                                1,
                                MATERIAL_LIBRARY,
                                "behavior_dummy",
                                coefficients,
                            )

                            result_dir = output_dir / "resu" / f"resu{suffix}"
                            script_file = output_dir / "script" / f"script{suffix}.sh"
                            write_AMITEX_launcher_ml(
                                str(script_file),
                                str(mesh_file),
                                str(zone_file),
                                str(algorithm_file),
                                str(material_file),
                                str(load_file),
                                str(result_dir),
                                env_amitex=AMITEX_ENV,
                                exe_amitex=AMITEX_EXECUTABLE,
                                cpus=CPU_COUNT,
                                prefix="vmacro",
                            )
                            script_file.chmod(0o755)


def run_simulations(output_dir=OUTPUT_DIR):
    """Execute the generated AMITEX launch scripts."""
    script_dir = Path(output_dir) / "script"
    subprocess.run(
        ["run-parts", "--regex", ".*sh$", str(script_dir)],
        check=True,
    )


def postprocess_results(output_dir=OUTPUT_DIR):
    """Convert velocity fields and collate the permeability tensors."""
    from .IOfcts import saveField2VTK

    results_dir = Path(output_dir) / "resu"
    bad_simulations = []
    directions = (1, 2, 3)

    for result_dir in sorted(results_dir.iterdir()):
        if not result_dir.is_dir():
            continue

        vtk_files = list(result_dir.glob("*.vtk"))
        if not vtk_files:
            if not list(result_dir.glob("*.vti")):
                bad_simulations.append(result_dir.name)
            continue

        converted_all_directions = True
        for direction in directions:
            prefix = result_dir / f"vmacro_dir{direction}"
            component_files = [
                Path(f"{prefix}_{component}.vtk") for component in ("x", "y", "z")
            ]
            if not all(path.is_file() for path in component_files):
                converted_all_directions = False
                continue

            velocity, origin, spacing = postProcBrinkman_readVFIELD(str(prefix))
            saveField2VTK(
                f"{prefix}_velocity.vti",
                velocity,
                "velocity",
                origin=origin,
                spacing=spacing,
            )

        if converted_all_directions:
            for vtk_file in vtk_files:
                vtk_file.unlink()
        else:
            bad_simulations.append(result_dir.name)

    tensor_file = results_dir / "Ktensor_all.csv"
    number_pattern = re.compile(r"-?\s*[0-9]+\.?[0-9]*(?:[Ee]\s*-?\s*[0-9]+)?")
    with tensor_file.open("w", encoding="utf-8") as output:
        output.write(
            "filename K11 K22 K33 K12 K13 K23 K21 K31 K32\n"
        )
        for result_dir in sorted(results_dir.iterdir()):
            if not result_dir.is_dir():
                continue

            log_file = result_dir / "vmacro.log"
            if not log_file.is_file():
                bad_simulations.append(result_dir.name)
                continue

            tensor = np.zeros((3, 3))
            with log_file.open(encoding="utf-8") as log:
                for line in log:
                    if "Ktensor:" not in line:
                        continue
                    for column in range(3):
                        tensor[:, column] = [
                            float(value)
                            for value in re.findall(number_pattern, next(log, ""))
                        ]

            values = (
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[0, 1],
                tensor[0, 2],
                tensor[1, 2],
                tensor[1, 0],
                tensor[2, 0],
                tensor[2, 1],
            )
            output.write(
                f"{result_dir.name} " + " ".join(str(value) for value in values) + "\n"
            )

    print(f"Simulations requiring attention: {bad_simulations}")
    print(f"Count: {len(bad_simulations)}")


def main():
    generate_inputs()
    run_simulations()
    postprocess_results()


if __name__ == "__main__":
    main()
