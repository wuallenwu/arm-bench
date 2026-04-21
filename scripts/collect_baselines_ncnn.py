"""
scripts/collect_baselines_ncnn.py — Collect baseline timings for ncnn starter kernels.

Compiles each starter .cpp file (from starter/problems.json) into TWO binaries:
  1. C-base binary  — uses the unmodified ncnn/mapped/{kernel}/{kernel}.cpp
  2. ARM binary    — uses the ARM-optimized ncnn/mapped/{kernel}/{kernel}_arm.cpp

Times each binary n times and records to baselines/ncnn.json:
  { "conv2d": { "scalar_ms": 123.4, "autovec_ms": 45.6 }, ... }

Where:
  scalar_ms  = time to run all test cases once with the C base implementation
  autovec_ms = time to run all test cases once with the ARM-optimized implementation

These are used by NCNNTools.submit() to compute speedup levels (level 2 = beats C
base, level 3 = beats ARM baseline).

Usage (from arm-bench/):
    # Provision a remote instance first (or reuse running one):
    python eval/provision.py --instance c7g.large

    # Then sync ncnn codebase and collect baselines:
    python -m scripts.collect_baselines_ncnn --isa neon --n 10

    # Or specify a single problem:
    python -m scripts.collect_baselines_ncnn --problem conv2d --isa neon
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# Add parent to sys.path so we can import eval.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.config import REPO_ROOT, NCNN_PROBLEMS_JSON, BASELINES_DIR
from eval.provision import get_running_instance, provision_ncnn

STARTER_DIR = REPO_ROOT / "starter"
DEFAULT_N = 10
DEFAULT_ISA = "neon"

# Injection markers (same as in NCNNTools)
_CANDIDATE_START = "// CANDIDATE_INJECT_START"
_CANDIDATE_END = "// CANDIDATE_INJECT_END"
_BASELINE_START = "// BASELINE_INJECT_START"
_BASELINE_END = "// BASELINE_INJECT_END"
_CANDIDATE_TC_START = "// CANDIDATE_TESTCASE_START"
_CANDIDATE_TC_END = "// CANDIDATE_TESTCASE_END"
_BASELINE_TC_START = "// BASELINE_TESTCASE_START"
_BASELINE_TC_END = "// BASELINE_TESTCASE_END"

# cmake libs needed per starter file
_STARTER_DEPS = {
    "convolution.cpp":            ["ncnn_stub", "mapped_conv_arm", "mapped_conv_base"],
    "convolution1d.cpp":          ["ncnn_stub", "mapped_conv_arm", "mapped_conv_base"],
    "convolutiondepthwise.cpp":   ["ncnn_stub", "mapped_conv_arm", "mapped_conv_base"],
    "deconvolution.cpp":          ["ncnn_stub", "mapped_conv_arm", "mapped_conv_base"],
    "deconvolutiondepthwise.cpp": ["ncnn_stub", "mapped_conv_arm", "mapped_conv_base"],
}
_EXTRA_INCLUDES = {
    "convolution.cpp":            ["arm-heavy-optimized/conv"],
    "convolution1d.cpp":          ["arm-heavy-optimized/conv"],
    "convolutiondepthwise.cpp":   ["arm-heavy-optimized/conv"],
    "deconvolution.cpp":          ["arm-heavy-optimized/conv"],
    "deconvolutiondepthwise.cpp": ["arm-heavy-optimized/conv"],
}

REMOTE_NCNN_ROOT = "~/ncnn"
REMOTE_STARTER_DIR = "~/simd-loops/starter"
REMOTE_BUILD_DIR = "~/simd-loops/starter/build_baselines"
REMOTE_CMAKE_BUILD = "~/ncnn/mapped/tests/build"


def _run(handle, cmd: str, timeout: int = 120):
    return handle.run(cmd, timeout=timeout)


def _upload(handle, local_path: str, remote_path: str):
    handle.upload_file(local_path, remote_path)


def _strip_block(source: str, start: str, end: str) -> str:
    import re
    return re.sub(re.escape(start) + ".*?" + re.escape(end), "", source, flags=re.DOTALL)


def _extract_test_functions(source: str, start: str, end: str) -> list:
    import re
    m = re.search(re.escape(start) + r"(.*?)" + re.escape(end), source, flags=re.DOTALL)
    if not m:
        return []
    return re.findall(r"void\s+(test_\w+)\s*\(\s*\)", m.group(1))


def _generate_main(test_funcs: list, suite_name: str) -> str:
    lines = ["\n// ── Auto-generated main ─────────────────────────────────────"]
    for fn in test_funcs:
        lines.append(f"void {fn}();")
    lines.append("")
    lines.append("int main() {")
    for fn in test_funcs:
        lines.append(f"    RUN_TEST({fn});")
    lines.append(f'    print_summary("{suite_name}");')
    lines.append("    return g_failed ? 1 : 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def setup_cmake(handle, isa: str) -> bool:
    """Configure cmake (once). Returns True on success."""
    march = {
        "neon": "armv8.2-a+fp16+dotprod",
        "sve":  "armv8.2-a+fp16+dotprod+sve",
        "sve2": "armv8.2-a+fp16+dotprod+sve2",
    }.get(isa, "armv8.2-a+fp16+dotprod")

    cmd = (
        f"mkdir -p {REMOTE_CMAKE_BUILD} && "
        f"cd {REMOTE_CMAKE_BUILD} && "
        f"cmake .. -DCMAKE_BUILD_TYPE=Release "
        f"-DCMAKE_CXX_FLAGS='-march={march}' 2>&1"
    )
    rc, output, _ = _run(handle, cmd, timeout=60)
    if rc != 0:
        print(f"    cmake configure failed:\n{output[:300]}")
        return False
    return True


def build_libs(handle, cmake_libs: list) -> bool:
    """Build cmake libs. Returns True on success."""
    targets = " ".join(f"--target {t}" for t in cmake_libs)
    cmd = f"cmake --build {REMOTE_CMAKE_BUILD} {targets} -j$(nproc) 2>&1"
    rc, output, _ = _run(handle, cmd, timeout=180)
    if rc != 0:
        print(f"    cmake build failed:\n{output[:300]}")
        return False
    return True


def compile_binary(handle, isa: str, starter_remote: str,
                   cmake_libs: list, extra_inc_dirs: list,
                   output_bin: str) -> tuple:
    """Compile starter_remote → output_bin. Returns (success, output)."""
    march = {
        "neon": "armv8.2-a+fp16+dotprod",
        "sve":  "armv8.2-a+fp16+dotprod+sve",
        "sve2": "armv8.2-a+fp16+dotprod+sve2",
    }.get(isa, "armv8.2-a+fp16+dotprod")

    include_flags = (
        f"-I {REMOTE_STARTER_DIR} "
        f"-I {REMOTE_NCNN_ROOT} "
        f"-I {REMOTE_NCNN_ROOT}/framework"
    )
    for subdir in extra_inc_dirs:
        include_flags += f" -I {REMOTE_NCNN_ROOT}/{subdir}"

    lib_flags = " ".join(
        f"{REMOTE_CMAKE_BUILD}/lib{t}.a" for t in cmake_libs
    )

    cmd = (
        f"mkdir -p {REMOTE_BUILD_DIR} && "
        f"clang++ -O2 -std=c++14 -march={march} -fopenmp "
        f"{include_flags} "
        f"{starter_remote} "
        f"{lib_flags} -lm -lstdc++ "
        f"-o {output_bin} 2>&1"
    )
    rc, output, _ = _run(handle, cmd, timeout=120)
    return rc == 0, output


def time_binary(handle, binary: str, n: int) -> float | None:
    """Run binary n times and return ms per invocation, or None on failure."""
    cmd = (
        f"t0=$(date +%s%N); "
        f"for i in $(seq 1 {n}); do {binary}; rc=$?; "
        f"if [ $rc -ne 0 ]; then exit $rc; fi; done; "
        f"t1=$(date +%s%N); "
        f'echo "TIME_NS=$((t1-t0))"'
    )
    rc, stdout, _ = _run(handle, cmd, timeout=600)
    if rc != 0:
        return None
    m = re.search(r"TIME_NS=(\d+)", stdout)
    if m:
        return round(int(m.group(1)) / 1e6 / n, 3)
    return None


def collect_one(handle, problem: dict, isa: str, n: int) -> dict | None:
    """
    Compile and time both C-base and ARM binaries for one problem.
    Returns a dict with scalar_ms and autovec_ms, or None on failure.
    """
    starter_file = problem["starter"]
    stem = starter_file.rsplit(".", 1)[0]

    cmake_libs = _STARTER_DEPS.get(starter_file, ["ncnn_stub", "mapped_conv_base"])
    extra_inc_dirs = _EXTRA_INCLUDES.get(starter_file, [])

    local_starter = STARTER_DIR / starter_file
    source = local_starter.read_text()

    results = {}

    for variant, (tc_start, tc_end, strip_start, strip_end) in {
        "candidate": (_CANDIDATE_TC_START, _CANDIDATE_TC_END, _BASELINE_START, _BASELINE_END),
        "baseline":  (_BASELINE_TC_START, _BASELINE_TC_END, _CANDIDATE_START, _CANDIDATE_END),
    }.items():
        # Generate variant source
        variant_src = _strip_block(source, strip_start, strip_end)
        # Strip the opposite testcase block too
        opp_tc_start = _BASELINE_TC_START if variant == "candidate" else _CANDIDATE_TC_START
        opp_tc_end   = _BASELINE_TC_END   if variant == "candidate" else _CANDIDATE_TC_END
        variant_src = _strip_block(variant_src, opp_tc_start, opp_tc_end)

        test_funcs = _extract_test_functions(source, tc_start, tc_end)
        if not test_funcs:
            print(f"    [{variant}] no test functions found — skipping")
            continue
        variant_src += _generate_main(test_funcs, f"{stem}_{variant}")

        # Upload
        remote_src = f"{REMOTE_STARTER_DIR}/{stem}_{variant}.cpp"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
            f.write(variant_src)
            tmp = f.name
        try:
            _upload(handle, tmp, remote_src)
        finally:
            os.unlink(tmp)

        # Compile
        output_bin = f"{REMOTE_BUILD_DIR}/{stem}_{variant}"
        ok, cxx_out = compile_binary(handle, isa, remote_src, cmake_libs, extra_inc_dirs, output_bin)
        if not ok:
            print(f"    [{variant}] compile failed: {cxx_out[:200]}")
            continue

        # Correctness check
        rc, stdout, _ = _run(handle, output_bin, timeout=60)
        if rc != 0 or "FAIL" in stdout:
            print(f"    [{variant}] correctness check failed")
            continue

        # Timing
        ms = time_binary(handle, output_bin, n)
        if ms is None:
            print(f"    [{variant}] timing failed")
            continue

        results[variant] = ms
        print(f"    [{variant}] {ms}ms/invocation")

    if not results:
        return None

    return {
        "scalar_ms":  results.get("candidate"),  # C base
        "autovec_ms": results.get("baseline"),   # ARM optimized
        "ref_ms":     None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect C-base and ARM baseline timings for ncnn starter kernels"
    )
    parser.add_argument("--problem", help="Single problem ID (default: all)")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Invocations per binary for timing (default: {DEFAULT_N})")
    parser.add_argument("--isa", default=DEFAULT_ISA, choices=["neon", "sve", "sve2"],
                        help=f"ISA tier (default: {DEFAULT_ISA})")
    parser.add_argument("--sync-ncnn", action="store_true",
                        help="Rsync ncnn codebase to remote before benchmarking")
    args = parser.parse_args()

    if not NCNN_PROBLEMS_JSON.exists():
        print(f"ERROR: {NCNN_PROBLEMS_JSON} not found.")
        sys.exit(1)

    problems_raw = json.loads(NCNN_PROBLEMS_JSON.read_text())
    if args.problem:
        problems_raw = [p for p in problems_raw if p["id"] == args.problem]
        if not problems_raw:
            print(f"Problem {args.problem!r} not found in starter/problems.json")
            sys.exit(1)

    handle = get_running_instance(args.isa)
    if handle is None:
        print(f"No running instance for ISA={args.isa}. "
              f"Run: python eval/provision.py --instance c7g.large")
        sys.exit(1)

    if args.sync_ncnn:
        provision_ncnn(handle)

    print(f"\nSetting up cmake on remote instance...")
    if not setup_cmake(handle, args.isa):
        print("cmake setup failed — aborting")
        sys.exit(1)

    print(f"\nBuilding ncnn libs...")
    all_libs = ["ncnn_stub", "mapped_conv_arm", "mapped_conv_base"]
    if not build_libs(handle, all_libs):
        print("lib build failed — aborting")
        sys.exit(1)

    print(f"\nCollecting baselines for {len(problems_raw)} kernels "
          f"(n={args.n}, isa={args.isa})")
    print(f"{'='*60}\n")

    results = {}
    for prob in problems_raw:
        pid = prob["id"]
        print(f"  [{pid}] {prob['starter']}")
        entry = collect_one(handle, prob, args.isa, args.n)
        if entry is None:
            print(f"    SKIPPED\n")
            continue
        results[pid] = entry
        print()

    # Merge with existing baselines
    BASELINES_DIR.mkdir(exist_ok=True)
    out_path = BASELINES_DIR / "ncnn.json"
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    existing.update(results)
    out_path.write_text(json.dumps(existing, indent=2))

    print(f"{'='*60}")
    print(f"Wrote {len(results)} entries to {out_path}\n")
    print(f"{'Problem':<20} {'C base ms':<15} {'ARM ms':<15}")
    print("-" * 50)
    for pid, entry in results.items():
        s = entry.get("scalar_ms", "FAIL")
        a = entry.get("autovec_ms", "N/A")
        print(f"{pid:<20} {str(s):<15} {str(a):<15}")


if __name__ == "__main__":
    main()
