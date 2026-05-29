# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Profile a CoreML (.mlpackage / .mlmodelc) model's inference latency and Apple Neural Engine (ANE) residency.

This is the measurement harness behind the YOLO iOS inference-performance work. It answers two questions for a
given model:

  1. How fast is it?      -> wall-clock predict() latency on the Neural Engine vs the CPU.
  2. Where does it run?   -> per-operation compute-device assignment (ANE / GPU / CPU), exposing ops that fall
                             back off the ANE, which is the usual cause of avoidable latency.

The Neural Engine path (``CPU_AND_NE``) is the engine that matters on iPhone; the Mac's ANE gives valid relative
comparisons between export/op variants even though absolute numbers differ from a specific A-series chip. Each
compute unit is timed in its own subprocess so a host GPU-compiler crash (a known macOS-only ``MPSGraph`` issue)
cannot take down the rest of the run.

Usage:
    python3 scripts/perf/coreml_profile.py path/to/model.mlpackage              # latency, all units
    python3 scripts/perf/coreml_profile.py path/to/model.mlpackage --plan       # + ANE residency
    python3 scripts/perf/coreml_profile.py path/to/model.mlpackage --runs 100   # more samples

Requires: coremltools>=8, numpy, pillow (macOS only — CoreML is not available on other platforms).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

UNITS = ("CPU_AND_NE", "CPU_ONLY", "ALL", "CPU_AND_GPU")


def _time_one_unit(path: str, unit: str, runs: int) -> dict | None:
    """Time ``runs`` predictions for one compute unit in an isolated subprocess; return None on a hard crash."""
    code = f"""
import sys, time, json, numpy as np, coremltools as ct
from PIL import Image
m = ct.models.MLModel({path!r}, compute_units=ct.ComputeUnit.{unit})
spec = m.get_spec()
feed = {{}}
for i in spec.description.input:
    t = i.type.WhichOneof("Type")
    if t == "imageType":
        w, h = i.type.imageType.width, i.type.imageType.height
        feed[i.name] = Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8))
    else:
        shape = [s if s > 0 else 1 for s in i.type.multiArrayType.shape]
        feed[i.name] = np.random.rand(*shape).astype(np.float32)
for _ in range(5):
    m.predict(feed)
ts = []
for _ in range({runs}):
    t = time.perf_counter(); m.predict(feed); ts.append((time.perf_counter() - t) * 1000)
ts.sort()
print(json.dumps({{"unit": {unit!r}, "runs": {runs}, "best": ts[0], "median": ts[len(ts)//2],
                   "p90": ts[int(len(ts)*0.9)], "mean": sum(ts)/len(ts)}}))
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return None


def _compute_plan(path: str, unit: str = "CPU_AND_NE") -> None:
    """Print per-device op counts and estimated cost share for the model (requires a compiled model)."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    compiled = ct.utils.compile_model(path) if Path(path).suffix == ".mlpackage" else path
    plan = MLComputePlan.load_from_path(compiled, compute_units=getattr(ct.ComputeUnit, unit))
    fn = plan.model_structure.program.functions["main"]
    dev_ops: Counter = Counter()
    dev_cost: Counter = Counter()
    cpu_types: Counter = Counter()
    for op in fn.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        device = type(usage.preferred_compute_device).__name__ if usage else "None"
        dev_ops[device] += 1
        est = plan.get_estimated_cost_for_mlprogram_operation(op)
        if est:
            dev_cost[device] += est.weight
        if device == "MLCPUComputeDevice":
            cpu_types[op.operator_name] += 1

    def short(name: str) -> str:
        return (
            name.replace("MLNeuralEngineComputeDevice", "ANE")
            .replace("MLCPUComputeDevice", "CPU")
            .replace("MLGPUComputeDevice", "GPU")
        )

    print(f"\nANE residency ({unit}):")
    print("  ops by device: ", {short(k): v for k, v in dev_ops.items()})
    print("  cost share:    ", {short(k): round(v, 3) for k, v in dev_cost.items()})
    if cpu_types:
        print("  CPU-bound ops: ", dict(cpu_types))


def main() -> None:
    """Parse arguments and run latency profiling (and optionally ANE residency analysis) for a CoreML model."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to .mlpackage or .mlmodelc")
    ap.add_argument("--runs", type=int, default=50, help="timed predictions per compute unit (default: 50)")
    ap.add_argument("--units", nargs="+", default=["CPU_AND_NE", "CPU_ONLY"], choices=UNITS)
    ap.add_argument("--plan", action="store_true", help="also print per-op ANE/CPU/GPU device assignment")
    args = ap.parse_args()

    path = str(Path(args.model).resolve())
    print(f"model: {path}")
    print(f"{'unit':12s} {'best':>8s} {'median':>8s} {'p90':>8s} {'mean':>8s}  (ms)")
    for unit in args.units:
        r = _time_one_unit(path, unit, args.runs)
        if r is None:
            print(f"{unit:12s} {'CRASH (host compiler issue; ignore for on-device)':>8s}")
        else:
            print(f"{unit:12s} {r['best']:8.2f} {r['median']:8.2f} {r['p90']:8.2f} {r['mean']:8.2f}")

    if args.plan:
        _compute_plan(path)


if __name__ == "__main__":
    main()
