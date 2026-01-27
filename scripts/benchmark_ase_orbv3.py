# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pip-system-certs",
#     "ase",
#     "orb-models",
#     "torch==2.9.1",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# matbench_speed = { path = "../." }
# ///

from matbench_speed.benchmark import benchmark
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

# Append "cuml-cu12==25.2.*" to use cuml-accelerated graph construction
calculators = {
    "Orb-v3-cons-inf-omat-high-cuml": 
        ORBCalculator(
            pretrained.orb_v3_conservative_inf_omat(
                device="cuda",
                precision="float32-high", # Keeping TF32 off for the benchmark for now
        ),
        device="cuda")
}
benchmark(calculators, atom_name="Si", lattice_constant=5.43)
