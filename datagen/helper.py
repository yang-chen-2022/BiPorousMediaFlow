"""Helpers for processing AMITEX output."""

import numpy as np


def postProcBrinkman_readVFIELD(prefix_output):
    """Read the three AMITEX velocity components into one vector field."""
    from .IOfcts import vtkFieldReader

    filenames = [f"{prefix_output}_{axis}.vtk" for axis in ("x", "y", "z")]
    field_names = [f"velocity_{axis}" for axis in ("x", "y", "z")]

    components = []
    for filename, field_name in zip(filenames, field_names):
        component, origin, spacing = vtkFieldReader(filename, field_name)
        components.append(component)

    velocity_field = np.moveaxis(np.asarray(components), 0, -1)
    return velocity_field, origin, spacing
