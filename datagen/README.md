# Data generation utilities

This folder contains Python utilities for creating random permeability fields,
preparing AMITEX input files, processing AMITEX output, and handling VTK data.

## Files

- `fft_ma.py` generates 2D and 3D stationary Gaussian random fields.
- `IOfcts.py` reads and writes VTK data, AMITEX XML files, and AMITEX binary
  coefficient files.
- `helper.py` reads AMITEX velocity components and combines them into a vector
  field.
- `datagen.py` generates the parameter sweep, runs AMITEX, converts
  velocity fields, and collates permeability tensors.
- `visualize_fields.py` displays `log10(ks)`, `log10(mu / ks)`, and
  solid-region `log10(beta)` for a generated sample.

## Requirements

Running the complete workflow also requires AMITEX (https://amitexfftp.github.io/AMITEX/index.html), with the BRINKMAN extension (link to be provided),
`run-parts`, and `mpirun`, with a working AMITEX environment script and material
library.

## Running the parameter sweep

Run the module from the directory containing `datagen`:

```bash
python -m datagen.datagen
```

The script uses site-specific defaults for its output directory and AMITEX
installation. Override them with environment variables before running:

```bash
export DATAGEN_OUTPUT_DIR=/path/to/output
export AMITEX_ENV=/path/to/amitex_fftp/env_amitex.sh
export AMITEX_EXECUTABLE=/path/to/amitex_fftp
export AMITEX_MATERIAL_LIBRARY=/path/to/libUmatAmitex.so
python -m datagen.datagen
```

The default sweep contains 800 samples. The script creates the required
`mesh/`, `zoneID/`, `mate/`, `script/`, and `resu/` directories, writes AMITEX
inputs, runs the generated scripts, converts complete velocity fields to VTI,
and writes `resu/Ktensor_all.csv`. The table is whitespace-delimited despite
its `.csv` extension.

After a result's velocity fields for all three loading directions have been
converted successfully, its source VTK files are removed. Keep a separate copy
of raw AMITEX output if you need those files later.

## Visualization demos

To generate and display the three permeability-related fields (requires
Matplotlib):

```bash
python -m datagen.visualize_fields
```

Edit the constants near the top of `visualize_fields.py` to change the sample
resolution, statistical parameters, viscosity, or fluid threshold.

To display example 2D and 3D Gaussian random fields:

```bash
python -m datagen.fft_ma
```
