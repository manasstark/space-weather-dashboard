"""Space Weather Physics Engine — the project's single scientific source
of truth for every derived space weather quantity.

Modules:
- core.py           VBz, Ey, Dynamic Pressure, Southward Duration family
- geometry.py        Clock Angle, Clock Angle Rate, Magnetic Shear, IMF Rotation Rate
- coupling.py        Newell Coupling Function, Akasofu epsilon, Boyle Index, Integrated Energy Input
- plasma.py           Magnetic/Thermal/Total Pressure, Plasma Beta, Alfven Speed/Mach Number
- magnetosphere.py    Magnetopause Stand-off Distance, Estimated Compression
- persistence.py      Solar Wind/Bt/Bz Persistence, IMF Persistence
- registry.py         Generalized opt-in feature registry resolver
- validation.py       Formula reference checks and NaN-safety checks

Production safety contract: swdss.models.features (production's own
contract file) delegates to core.py for VBz/Ey/Dynamic Pressure only —
the three quantities production's trained models depend on — and that
delegation was verified byte-identical against the pre-migration inline
implementation across every production dataset before being wired in.
Every other quantity in this package is consumed by the research
laboratories, the dashboard, and the narrative interpretation module;
none of it is on the production training/inference path.

Six formula inconsistencies the architectural audit found across the
IMF/Kp/AE Research Laboratories (Plasma Beta, Newell Coupling Function,
Akasofu epsilon, Boyle Index, Magnetic Shear, Integrated Energy Input)
were each resolved to one scientifically-justified canonical
implementation here — see coupling.py, plasma.py, and geometry.py's
module docstrings for the specific reasoning and reference behind each
resolution, and which lab's feature values changed as a result.
"""

from swdss.physics.core import (
    add_derived_physics_features,
    dynamic_pressure_scalar,
    dynamic_pressure_series,
    ey_scalar,
    ey_series,
    integrated_ey_series,
    integrated_southward_bz_series,
    integrated_vbz_series,
    southward_duration_series,
    strong_southward_duration_series,
    vbz_scalar,
    vbz_series,
)
from swdss.physics.coupling import (
    akasofu_epsilon_series,
    boyle_index_series,
    integrated_energy_input_series,
    newell_coupling_series,
)
from swdss.physics.geometry import (
    clock_angle_rate_series,
    clock_angle_scalar,
    clock_angle_series,
    imf_rotation_rate_series,
    magnetic_shear_series,
)
from swdss.physics.magnetosphere import estimated_compression_series, magnetopause_standoff_series
from swdss.physics.persistence import imf_persistence_series, persistence_stats_series
from swdss.physics.plasma import (
    alfven_mach_number_series,
    alfven_speed_series,
    magnetic_pressure_series,
    plasma_beta_series,
    thermal_pressure_series,
    total_pressure_series,
)

__all__ = [
    "add_derived_physics_features",
    "dynamic_pressure_scalar",
    "dynamic_pressure_series",
    "ey_scalar",
    "ey_series",
    "integrated_ey_series",
    "integrated_southward_bz_series",
    "integrated_vbz_series",
    "southward_duration_series",
    "strong_southward_duration_series",
    "vbz_scalar",
    "vbz_series",
    "akasofu_epsilon_series",
    "boyle_index_series",
    "integrated_energy_input_series",
    "newell_coupling_series",
    "clock_angle_rate_series",
    "clock_angle_scalar",
    "clock_angle_series",
    "imf_rotation_rate_series",
    "magnetic_shear_series",
    "estimated_compression_series",
    "magnetopause_standoff_series",
    "imf_persistence_series",
    "persistence_stats_series",
    "alfven_mach_number_series",
    "alfven_speed_series",
    "magnetic_pressure_series",
    "plasma_beta_series",
    "thermal_pressure_series",
    "total_pressure_series",
]
