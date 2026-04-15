"""
eval/tools.py — Tool implementations for the simd-loops LLM benchmark.

Each function executes via SSH on a provisioned Arm EC2 instance.
These are the tools exposed to the LLM agent (compile, run, perf, disassemble, submit).

The agent writes C code only — it never sees SSH commands or bash.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from eval.config import ISA_MAKE_TARGET, REPO_ROOT, load_baselines, load_problem_sizes, ISA_TIER
from eval.provision import InstanceHandle

CANDIDATE_START = "// CANDIDATE_INJECT_START"
CANDIDATE_END = "// CANDIDATE_INJECT_END"


@dataclass
class CompileResult:
    success: bool
    errors: str = ""
    warnings: str = ""

    def to_tool_result(self) -> dict:
        if self.success:
            return {"success": True, "warnings": self.warnings or "(none)"}
        return {"success": False, "errors": self.errors}


@dataclass
class RunResult:
    correct: bool
    runtime_ms: float | None = None
    output: str = ""

    def to_tool_result(self) -> dict:
        return {
            "correct": self.correct,
            "runtime_ms": self.runtime_ms,
            "output": self.output.strip(),
        }


@dataclass
class PerfResult:
    cycles: int | None = None
    instructions: int | None = None
    ipc: float | None = None
    cache_misses_per_iter: float | None = None  # LLC misses divided by n iterations
    task_clock_ms: float | None = None  # on-CPU ms per iteration (excludes scheduler idle)
    raw_output: str = ""

    def to_tool_result(self) -> dict:
        return {
            "cycles": self.cycles,
            "instructions": self.instructions,
            "ipc": self.ipc,
            "cache_misses_per_iter": self.cache_misses_per_iter,
            "task_clock_ms": self.task_clock_ms,
            "raw_output": self.raw_output.strip(),
        }


@dataclass
class DisasmResult:
    asm: str = ""
    bytes: int = 0

    def to_tool_result(self) -> dict:
        return {"asm": self.asm, "bytes": self.bytes}


@dataclass
class EvalResult:
    correct: bool
    speedup_vs_scalar: float | None = None
    speedup_vs_autovec: float | None = None
    speedup_vs_ref: float | None = None
    level: int = 0
    compile_error: str = ""
    runtime_ms: float | None = None
    tool_calls: int = 0
    explanation: str = ""   # agent's reasoning about its approach (from submit)
    # Timing at each PERF_SIZE: {size: runtime_ms}. Populated at submit time.
    perf_by_size: dict | None = None
    # ISO-8601 timestamp of when this result was produced.
    timestamp: str = ""
    # Per-turn trace: list of {turn, tool, reasoning, args_summary, result_summary}.
    trace: list = field(default_factory=list)
    # All code versions attempted during the session (full code + metrics).
    # Excluded from to_dict() to keep results JSON small; saved separately in traces/.
    code_versions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "speedup_vs_scalar": self.speedup_vs_scalar,
            "speedup_vs_autovec": self.speedup_vs_autovec,
            "speedup_vs_ref": self.speedup_vs_ref,
            "level": self.level,
            "compile_error": self.compile_error,
            "runtime_ms": self.runtime_ms,
            "tool_calls": self.tool_calls,
            "explanation": self.explanation,
            "perf_by_size": self.perf_by_size,
            "timestamp": self.timestamp,
            "trace": self.trace,
        }


class SIMDTools:
    """
    SSH-backed tools for compiling and running SIMD kernels on a remote Arm instance.

    Used both by the agentic eval loop (as LLM tool calls) and by the
    single-shot eval harness (eval_from_generations.py).
    """

    def __init__(self, handle: InstanceHandle, problem_id: str, isa: str):
        self.handle = handle
        self.problem_id = problem_id
        self.isa = isa
        self.loop_num = problem_id.split("_")[1]   # "loop_001" → "001"
        self.make_target = ISA_MAKE_TARGET[isa]
        self._last_compile_ok = False
        self._binary_exists = False   # True once any compile has ever succeeded this session
        self._tool_calls = 0
        self._last_candidate_code: str | None = None  # stored for size-specific recompiles
        self._default_size: int | None = None          # parsed once from source on first use

        # Remote paths (on the instance)
        self.remote_root = "~/simd-loops"
        self.remote_loop_file = f"{self.remote_root}/loops/loop_{self.loop_num}.c"
        self.remote_binary = (
            f"{self.remote_root}/build/{self.make_target}/bin/simd_loops"
        )

        # Per-problem test sizes (loaded directly from problem.py, not the full index)
        # Pass isa so c8g targets use PERF_SIZES_C8G when defined.
        self._edge_sizes, self._perf_sizes = load_problem_sizes(problem_id, isa=isa)

        # Restore the remote loop file to the clean local stub so any candidate
        # code left by a previous session doesn't pollute this session's compiles.
        self._restore_stub()

    def _restore_stub(self):
        """Rsync all local loop files to remote, clearing any injected candidate code."""
        self.handle.rsync_to(
            str(REPO_ROOT / "loops"),
            f"{self.remote_root}/loops",
            excludes=["*.o", "build"],
        )

    # ─── Tool: compile ───────────────────────────────────────────────────────

    def compile(self, code: str) -> CompileResult:
        """
        Inject `code` as the HAVE_CANDIDATE implementation and compile.

        Args:
            code: C function body for inner_loop_NNN. Must preserve the
                  existing function signature.

        Returns:
            CompileResult with success flag and any errors/warnings.
        """
        self._tool_calls += 1

        # Catch history-compression placeholders before wasting an SSH round-trip
        if "[prior successful attempt:" in code and "omitted" in code:
            return CompileResult(
                success=False,
                errors=(
                    "You submitted a history-compression placeholder, not real code. "
                    "The prior binary is still compiled on the remote — call run() or perf() "
                    "to test it, or write a fresh SVE implementation to compile()."
                ),
            )

        # 1. Patch the source file locally and upload it
        local_loop_file = REPO_ROOT / "loops" / f"loop_{self.loop_num}.c"
        source = local_loop_file.read_text()

        if CANDIDATE_START not in source:
            return CompileResult(
                success=False,
                errors=f"CANDIDATE_INJECT_START marker missing from loop_{self.loop_num}.c. "
                       f"Run: python scripts/extract_dataset.py --add-candidate-blocks",
            )

        new_block = f"{CANDIDATE_START}\n{code}\n{CANDIDATE_END}"
        patched = re.sub(
            re.escape(CANDIDATE_START) + ".*?" + re.escape(CANDIDATE_END),
            new_block,
            source,
            flags=re.DOTALL,
        )

        # Write to a temp file and upload
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write(patched)
            tmp_path = f.name
        try:
            self.handle.upload_file(tmp_path, self.remote_loop_file)
        finally:
            os.unlink(tmp_path)

        # 2. Compile with HAVE_CANDIDATE flag
        # Remove stale object/symlink for this loop so make doesn't fail on
        # "ln: File exists" when the previous build was interrupted mid-way.
        obj_base = f"{self.remote_root}/build/{self.make_target}/_obj"
        make_cmd = (
            f"cd {self.remote_root} && "
            f"rm -f {obj_base}/loops/loop_{self.loop_num}.o "
            f"       {obj_base}/_lnk/loop_{self.loop_num}.o && "
            f"make {self.make_target} "
            f"EXTRA_FLAGS='-DHAVE_CANDIDATE' "
            f"2>&1"
        )
        rc, combined, _ = self.handle.run(make_cmd, timeout=120)

        warnings = "\n".join(
            l for l in combined.splitlines()
            if "warning:" in l.lower() and "error:" not in l.lower()
        )
        errors = "\n".join(
            l for l in combined.splitlines()
            if "error:" in l.lower()
        )

        if rc != 0:
            self._last_compile_ok = False
            return CompileResult(success=False, errors=errors or combined)

        self._last_compile_ok = True
        self._binary_exists = True
        self._last_candidate_code = code
        return CompileResult(success=True, warnings=warnings)

    # ─── Tool: run ───────────────────────────────────────────────────────────

    def run(self, n: int = 100, size: int | None = None) -> RunResult:
        """
        Run the compiled binary for this loop and report correctness + timing.

        Args:
            n:    Number of iterations (more = more stable timing, slower).
            size: If given, recompile at this input size before running, then
                  restore the default-size binary afterwards. Useful for testing
                  correctness and timing at non-default array lengths.

        Returns:
            RunResult with correct flag and runtime_ms.
        """
        self._tool_calls += 1
        if not self._binary_exists:
            return RunResult(correct=False, output="No compiled binary — run compile() first.")

        if size is not None:
            return self._run_at_explicit_size(n, size)

        loop_hex = int(self.loop_num)
        time_cmd = (
            f"t0=$(date +%s%N); "
            f"{self.remote_binary} -k {loop_hex} -n {n}; "
            f"rc=$?; "
            f"t1=$(date +%s%N); "
            f'echo "TIME_NS=$((t1-t0))"; '
            f"exit $rc"
        )
        rc, stdout, stderr = self.handle.run(time_cmd, timeout=300)
        correct = "Checksum correct." in stdout

        runtime_ms = None
        m = re.search(r"TIME_NS=(\d+)", stdout)
        if m:
            total_ns = int(m.group(1))
            runtime_ms = round(total_ns / 1e6, 3)

        output_clean = stdout.replace(f"TIME_NS={m.group(0) if m else ''}", "").strip()
        return RunResult(correct=correct, runtime_ms=runtime_ms, output=output_clean)

    def _run_at_explicit_size(self, n: int, size: int) -> RunResult:
        """
        Compile candidate at `size`, run n iterations, restore default binary.

        Correctness is reported as "did not abort/crash" rather than matching
        the hardcoded checksum (which is only valid at the default compiled size).
        The binary's own output (LOOP_RESULT line) is included for context.
        """
        if self._last_candidate_code is None:
            return RunResult(correct=False, output="No candidate code stored — run compile() first.")

        bin_path = self._compile_at_size(size, have_candidate=True)
        if bin_path is None:
            return RunResult(correct=False, output=f"size={size}: compile failed")

        loop_decimal = int(self.loop_num)
        time_cmd = (
            f"t0=$(date +%s%N); "
            f"{bin_path} -k {loop_decimal} -n {n}; "
            f"rc=$?; "
            f"t1=$(date +%s%N); "
            f'echo "TIME_NS=$((t1-t0))"; '
            f"exit $rc"
        )
        rc, stdout, _ = self.handle.run(time_cmd, timeout=300)

        # rc==0: correct checksum. rc==1: wrong checksum (expected at non-default SIZE).
        # rc==2 or "ABORT": explicit abort (no implementation / alloc fail).
        # Any other rc (e.g. 132=SIGILL, 139=SIGSEGV) means the binary crashed.
        ran_ok = rc in (0, 1) and "ABORT" not in stdout

        runtime_ms = None
        m = re.search(r"TIME_NS=(\d+)", stdout)
        if m:
            runtime_ms = round(int(m.group(1)) / 1e6, 3)

        output_clean = re.sub(r"TIME_NS=\d+", "", stdout).strip()

        # Restore default binary so subsequent run()/perf() calls work correctly
        self._compile_at_size(self._get_default_size() or size, have_candidate=True)

        return RunResult(correct=ran_ok, runtime_ms=runtime_ms, output=output_clean)

    # ─── Tool: perf ──────────────────────────────────────────────────────────

    def perf(self, n: int = 100, size: int | None = None) -> PerfResult:
        """
        Run perf stat to collect hardware PMU counters.

        Available on Graviton3/4 via Nitro:
          - cycles, instructions, IPC
          - r04 = L1D_CACHE accesses, r03 = L1D_CACHE_REFILL (misses)

        Note: L2/L3 counters are not exposed by the Nitro hypervisor.

        Args:
            n:    Iteration count.
            size: If given, recompile at this input size before running, then
                  restore the default-size binary afterwards.

        Returns:
            PerfResult with cycles, instructions, IPC, cache_misses_per_iter, and task_clock_ms
            (on-CPU milliseconds per iteration, excluding scheduler idle time).
        """
        self._tool_calls += 1
        if not self._binary_exists:
            return PerfResult(raw_output="No compiled binary — run compile() first.")

        if size is not None:
            return self._perf_at_explicit_size(n, size)

        loop_hex = int(self.loop_num)
        # The kernel-versioned perf binary under /usr/lib has PMU support on
        # Graviton.  Probe for the first one that exists; fall back to the
        # system 'perf' wrapper (which may warn but still work).
        perf_cmd = (
            f"PERF=$(ls /usr/lib/linux-aws-*-tools-*/perf 2>/dev/null | head -1); "
            f"PERF=${{PERF:-perf}}; "
            f"{self.remote_binary} -k {loop_hex} -n 1 >/dev/null 2>&1; "  # warmup
            f"sudo $PERF stat "
            f"-e cycles,instructions,cache-misses,task-clock "
            f"{self.remote_binary} -k {loop_hex} -n {n} "
            f"2>&1"
        )
        rc, output, _ = self.handle.run(perf_cmd, timeout=300)
        return self._parse_perf_output(output, n)

    def _perf_at_explicit_size(self, n: int, size: int) -> PerfResult:
        """Compile candidate at `size`, run perf stat, restore default binary."""
        if self._last_candidate_code is None:
            return PerfResult(raw_output="No candidate code stored — run compile() first.")

        bin_path = self._compile_at_size(size, have_candidate=True)
        if bin_path is None:
            return PerfResult(raw_output=f"size={size}: compile failed")

        loop_decimal = int(self.loop_num)
        perf_cmd = (
            f"PERF=$(ls /usr/lib/linux-aws-*-tools-*/perf 2>/dev/null | head -1); "
            f"PERF=${{PERF:-perf}}; "
            f"{bin_path} -k {loop_decimal} -n 1 >/dev/null 2>&1; "  # warmup
            f"sudo $PERF stat "
            f"-e cycles,instructions,cache-misses,task-clock "
            f"{bin_path} -k {loop_decimal} -n {n} "
            f"2>&1"
        )
        rc, output, _ = self.handle.run(perf_cmd, timeout=300)

        # Restore default binary
        self._compile_at_size(self._get_default_size() or size, have_candidate=True)

        return self._parse_perf_output(output, n)

    @staticmethod
    def _parse_perf_output(output: str, n: int = 1) -> PerfResult:
        cycles = _parse_perf_counter(output, "cycles")
        instructions = _parse_perf_counter(output, "instructions")

        ipc = None
        m = re.search(r"([\d.]+)\s+insn per cycle", output)
        if m:
            ipc = float(m.group(1))
        elif cycles and instructions:
            ipc = round(instructions / cycles, 2) if cycles > 0 else None

        raw_cache_misses = _parse_perf_counter(output, "cache-misses")
        cache_misses_per_iter = None
        if raw_cache_misses is not None and n > 0:
            cache_misses_per_iter = round(raw_cache_misses / n, 1)

        # task-clock: on-CPU time per iteration (excludes time sleeping/preempted).
        # Older perf: "   1,234.56 msec task-clock"  (value in ms)
        # Newer perf: "   1234567  task-clock"        (value in nanoseconds)
        task_clock_ms = None
        m = re.search(r"([\d,]+\.?\d*)\s+msec\s+task-clock", output)
        if m and n > 0:
            task_clock_ms = round(float(m.group(1).replace(",", "")) / n, 4)
        else:
            raw_ns = _parse_perf_counter(output, "task-clock")
            if raw_ns is not None and n > 0:
                task_clock_ms = round(raw_ns / n / 1e6, 4)

        return PerfResult(
            cycles=cycles,
            instructions=instructions,
            ipc=ipc,
            cache_misses_per_iter=cache_misses_per_iter,
            task_clock_ms=task_clock_ms,
            raw_output=output,
        )

    # ─── Tool: disassemble ───────────────────────────────────────────────────

    def disassemble(self, fn: str | None = None) -> DisasmResult:
        """
        Disassemble the compiled binary, optionally filtered to a function.

        Args:
            fn: Function name to filter to (e.g. "inner_loop_001").
                If None, returns the full disassembly (may be large).

        Returns:
            DisasmResult with assembly text and approximate byte count.
        """
        self._tool_calls += 1
        if not self._last_compile_ok:
            return DisasmResult(asm="No compiled binary — run compile() first.")

        def _objdump_fn(name: str) -> str:
            return (
                f"llvm-objdump-18 -d {self.remote_binary} "
                f"| awk '/<{name}>:/ {{p=1}} p && /<[a-zA-Z_].*>:/ && !/<{name}>:/ {{p=0}} p'"
            )

        if fn:
            rc, output, stderr = self.handle.run(_objdump_fn(fn), timeout=60)
            if rc != 0:
                return DisasmResult(asm=f"objdump failed: {stderr}")
            # If the requested symbol was inlined, fall back to the outer loop wrapper
            if not output.strip():
                fallback = f"loop_{self.loop_num}"
                rc, output, stderr = self.handle.run(_objdump_fn(fallback), timeout=60)
                if rc != 0:
                    return DisasmResult(asm=f"objdump failed: {stderr}")
        else:
            rc, output, stderr = self.handle.run(
                f"llvm-objdump-18 -d {self.remote_binary}", timeout=60
            )
            if rc != 0:
                return DisasmResult(asm=f"objdump failed: {stderr}")

        # Truncate to first 500 lines to avoid flooding context
        lines = output.splitlines()
        truncated = False
        if len(lines) > 500:
            lines = lines[:500]
            truncated = True
        asm = "\n".join(lines)
        if truncated:
            asm += "\n... (truncated at 500 lines)"

        return DisasmResult(asm=asm, bytes=len(output.encode()))

    # ─── Tool: submit ─────────────────────────────────────────────────────────

    def submit(self, code: str, explanation: str = "") -> EvalResult:
        """
        Final submission: compile, check correctness, score against baselines.

        Authoritative timing uses the largest PERF_SIZE (cache-busting, DRAM-bound)
        so that speedups are measured against stable, memory-bandwidth-limited runs.
        Falls back to the default compiled-in size if no PERF_SIZES are defined.

        Args:
            code: The optimized C implementation to submit.
            explanation: Agent's description of the approach and perf observations.

        Returns:
            EvalResult with correctness, speedup levels, final score, and per-size
            performance data (perf_by_size).
        """
        self._tool_calls += 1

        # Compile
        cr = self.compile(code)
        if not cr.success:
            return EvalResult(
                correct=False,
                level=0,
                compile_error=cr.errors,
                tool_calls=self._tool_calls,
            )

        # Run correctness check at default size
        rr = self.run(n=10)
        if not rr.correct:
            return EvalResult(
                correct=False,
                level=0,
                tool_calls=self._tool_calls,
            )

        # Edge-case correctness: sizes from per-problem EDGE_SIZES
        edge_fail = self._check_edge_sizes()
        if edge_fail:
            return EvalResult(
                correct=False,
                level=0,
                compile_error=f"Edge-case correctness failure: {edge_fail}",
                tool_calls=self._tool_calls,
            )

        # Performance at larger sizes: collect timing at each PERF_SIZE
        perf_by_size = self._collect_perf_sizes()

        # Authoritative timing: use the largest PERF_SIZE result (ms/iter at
        # cache-busting size). Falls back to 100 iters at the default compiled
        # size for loops that have no PERF_SIZES (fixed-dimension, etc.).
        runtime_ms: float | None = None
        if self._perf_sizes and perf_by_size:
            valid = {s: ms for s, ms in perf_by_size.items() if ms is not None}
            if valid:
                runtime_ms = valid[max(valid)]  # ms/iter at largest PERF_SIZE
        if runtime_ms is None:
            rr_final = self.run(n=100)
            runtime_ms = round(rr_final.runtime_ms / 100, 4) if rr_final.runtime_ms else None

        # Load baselines
        tier = ISA_TIER.get(self.isa, "c7g")
        baselines = load_baselines(tier)
        baseline = baselines.get(self.problem_id, {})
        scalar_ms = baseline.get("scalar_ms")
        autovec_ms = baseline.get("autovec_ms")
        ref_ms = baseline.get("ref_ms")  # hand-written SVE/SVE2/SME2 reference

        speedup_vs_scalar = None
        speedup_vs_autovec = None
        speedup_vs_ref = None
        level = 1  # correct

        if runtime_ms and scalar_ms:
            speedup_vs_scalar = round(scalar_ms / runtime_ms, 2)
            if speedup_vs_scalar > 1.0:
                level = 2

        if runtime_ms and autovec_ms:
            speedup_vs_autovec = round(autovec_ms / runtime_ms, 2)
            if level >= 2 and speedup_vs_autovec > 1.0:
                level = 3

        if runtime_ms and ref_ms:
            speedup_vs_ref = round(ref_ms / runtime_ms, 2)
            if level >= 3 and speedup_vs_ref > 1.0:
                level = 4  # beats hand-written SVE reference

        return EvalResult(
            correct=True,
            speedup_vs_scalar=speedup_vs_scalar,
            speedup_vs_autovec=speedup_vs_autovec,
            speedup_vs_ref=speedup_vs_ref,
            level=level,
            runtime_ms=runtime_ms,
            tool_calls=self._tool_calls,
            explanation=explanation,
            perf_by_size=perf_by_size if perf_by_size else None,
        )

    # ─── Edge-case and perf-size helpers ─────────────────────────────────────

    def _get_default_size(self) -> int | None:
        """Parse the default SIZE from the local loop source file (cached)."""
        if self._default_size is not None:
            return self._default_size
        local_loop_file = REPO_ROOT / "loops" / f"loop_{self.loop_num}.c"
        m = re.search(r"#define SIZE\s+(\d+)", local_loop_file.read_text())
        if m:
            self._default_size = int(m.group(1))
        return self._default_size

    def _run_at_size(self, binary: str, size: int) -> str | None:
        """
        Run `binary` compiled with -DSIZE=<size>, return the LOOP_RESULT value
        string from stdout, or None if the binary aborted.
        """
        loop_decimal = int(self.loop_num)
        cmd = f"{binary} -k {loop_decimal} -n 1 2>&1"
        rc, stdout, _ = self.handle.run(cmd, timeout=60)
        if rc == 2:  # ABORT exit code
            return None
        m = re.search(r"LOOP_RESULT:\s*(.+)", stdout)
        return m.group(1).strip() if m else None

    def _compile_at_size(self, size: int, have_candidate: bool) -> str | None:
        """
        Compile with -DSIZE=<size>. Returns the binary path on success, None on failure.
        Uses HAVE_CANDIDATE or HAVE_NATIVE depending on have_candidate flag.

        Strategy: first ensure the target binary exists with all *other* loops'
        .o files already built (a plain make with no custom EXTRA_FLAGS).  Then
        recompile only the specific loop .o at the requested size and relink.
        This avoids triggering a full rebuild where other loops may reject the
        custom SIZE via _Static_assert (e.g. loop_023 requires SIZE % 16 == 0).
        """
        if have_candidate:
            extra = f"-DHAVE_CANDIDATE -DSIZE={size}"
            target = self.make_target
            base_extra = "-DHAVE_CANDIDATE"
        else:
            extra = f"-DHAVE_NATIVE -U__ARM_NEON -U__ARM_FEATURE_SVE -U__ARM_FEATURE_SVE2 -U__ARM_FEATURE_SME -DSIZE={size}"
            target = "c-scalar"
            base_extra = "-DHAVE_NATIVE -U__ARM_NEON -U__ARM_FEATURE_SVE -U__ARM_FEATURE_SVE2 -U__ARM_FEATURE_SME"

        obj_dir = f"{self.remote_root}/build/{target}/_obj"
        loop_o  = f"{obj_dir}/loops/loop_{self.loop_num}.o"
        lnk_o   = f"{obj_dir}/_lnk/loop_{self.loop_num}.o"

        # Step 1: ensure all OTHER loops are already compiled (plain build,
        # no custom SIZE).  This is a no-op if the binary is already up to date.
        warmup_cmd = f"cd {self.remote_root} && make {target} EXTRA_FLAGS='{base_extra}' 2>&1"
        rc, _, _ = self.handle.run(warmup_cmd, timeout=120)
        if rc != 0:
            return None  # even the baseline build failed

        # Step 2: recompile only this loop's .o at the requested size, then relink.
        clean_cmd  = f"rm -f {loop_o} {lnk_o}"
        relink_cmd = (
            f"cd {self.remote_root} && {clean_cmd} && "
            f"make {target} EXTRA_FLAGS='{extra}' 2>&1"
        )
        rc, output, _ = self.handle.run(relink_cmd, timeout=120)
        if rc != 0:
            return None
        return f"{self.remote_root}/build/{target}/bin/simd_loops"

    def _check_edge_sizes(self) -> str | None:
        """
        Test correctness at each size in EDGE_SIZES (from problem metadata).

        For each size, compares the candidate result against the c-scalar reference.
        Returns an error string on the first failure, None if all pass.
        """
        if not self._edge_sizes:
            return None  # no edge sizes defined for this problem

        for size in self._edge_sizes:
            # Compile scalar reference at this size
            scalar_bin = self._compile_at_size(size, have_candidate=False)
            if scalar_bin is None:
                continue  # scalar build failed at this size (e.g. size too large), skip

            # Compile candidate at this size
            candidate_bin = self._compile_at_size(size, have_candidate=True)
            if candidate_bin is None:
                return f"size={size}: candidate failed to compile"

            scalar_result = self._run_at_size(scalar_bin, size)
            candidate_result = self._run_at_size(candidate_bin, size)

            if candidate_result is None:
                return f"size={size}: candidate aborted"

            if scalar_result is None:
                continue  # scalar aborted at this size, can't compare

            # Compare: parse as floats if possible, else string compare
            try:
                sv = float(scalar_result.split()[0])
                cv = float(candidate_result.split()[0])
                tolerance = max(abs(sv) * 1e-4, 1e-6)
                if abs(sv - cv) > tolerance:
                    return f"size={size}: scalar={sv:.6g} candidate={cv:.6g} (mismatch)"
            except (ValueError, IndexError):
                if scalar_result != candidate_result:
                    return f"size={size}: scalar={scalar_result!r} candidate={candidate_result!r} (mismatch)"

        # Restore default-size binary so run()/perf() work after submit
        default_size = self._get_default_size()
        if default_size is not None:
            self._compile_at_size(default_size, have_candidate=True)

        return None  # all edge cases passed

    def _collect_perf_sizes(self) -> dict[int, float | None]:
        """
        Run the candidate at each PERF_SIZE and collect timing (ms per iteration).
        Also verifies correctness at each large size.
        Returns {size: runtime_ms} for sizes that ran successfully.
        """
        if not self._perf_sizes:
            return {}

        results: dict[int, float | None] = {}
        loop_decimal = int(self.loop_num)

        for size in self._perf_sizes:
            bin_path = self._compile_at_size(size, have_candidate=True)
            if bin_path is None:
                results[size] = None
                continue

            # Quick sanity: ensure the binary doesn't abort at this size.
            # (Full correctness vs scalar is already covered by _check_edge_sizes;
            # the hardcoded checksum is only valid at the default SIZE.)
            result_str = self._run_at_size(bin_path, size)
            if result_str is None:
                results[size] = None  # aborted
                continue

            # Timing: 10 iterations (each ~10ms → ~0.1s total per size)
            time_cmd = (
                f"t0=$(date +%s%N); "
                f"{bin_path} -k {loop_decimal} -n 10; "
                f"rc=$?; "
                f"t1=$(date +%s%N); "
                f'echo "TIME_NS=$((t1-t0))"; '
                f"exit $rc"
            )
            rc, stdout, _ = self.handle.run(time_cmd, timeout=600)
            if rc == 2 or "ABORT" in stdout:
                results[size] = None  # aborted at this size
                continue

            m = re.search(r"TIME_NS=(\d+)", stdout)
            if m:
                total_ms = int(m.group(1)) / 1e6
                results[size] = round(total_ms / 10, 4)  # ms per iteration
            else:
                results[size] = None

        # Restore default-size binary
        default_size = self._get_default_size()
        if default_size is not None:
            self._compile_at_size(default_size, have_candidate=True)

        return results

    # ─── OpenAI-compatible tool schemas ──────────────────────────────────────

    @staticmethod
    def tool_schemas() -> list[dict]:
        """Return OpenAI-compatible function tool definitions for LiteLLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "compile",
                    "description": (
                        "Compile your SIMD implementation on the target Arm instance. "
                        "Returns whether compilation succeeded and any errors/warnings."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Your complete C implementation of the inner_loop function. "
                                    "Must preserve the exact function signature from the scalar version."
                                ),
                            },
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": (
                        "Run the last compiled binary and check correctness + timing. "
                        "Must call compile() successfully first. "
                        "Pass size= to test at a different input array length (recompiles automatically)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "integer",
                                "description": "Number of iterations (default 100; more = more stable timing).",
                                "default": 100,
                            },
                            "size": {
                                "type": "integer",
                                "description": (
                                    "Input array size to test (overrides the default compiled size). "
                                    "Useful for verifying edge cases (e.g. size=1) or large inputs."
                                ),
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "perf",
                    "description": (
                        "Run perf stat to collect hardware PMU counters: "
                        "cycles, instructions, IPC, cache_misses_per_iter (LLC misses per iteration), "
                        "and task_clock_ms (on-CPU ms per iteration — more precise than run() wall-clock). "
                        "Use size= to profile at a large input (e.g. size=500000) so data spills out of "
                        "cache and you measure real memory bandwidth pressure."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "integer",
                                "description": "Number of iterations.",
                                "default": 100,
                            },
                            "size": {
                                "type": "integer",
                                "description": (
                                    "Input array size to test (overrides the default compiled size). "
                                    "Useful for measuring performance at larger inputs."
                                ),
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "disassemble",
                    "description": (
                        "Disassemble the compiled binary. Filter to a specific function "
                        "to see the generated AArch64 instructions. Useful for checking "
                        "whether the compiler vectorized correctly."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fn": {
                                "type": "string",
                                "description": (
                                    "Function name to filter to, e.g. 'inner_loop_001'. "
                                    "If omitted, returns full disassembly (may be large)."
                                ),
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit",
                    "description": (
                        "Submit your final implementation for scoring. "
                        "Compiles, runs at large DRAM-bound sizes, and computes speedup vs scalar and autovec baselines. "
                        "Call this when you have a correct, profiled implementation you are satisfied with."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Your final optimized C implementation.",
                            },
                            "explanation": {
                                "type": "string",
                                "description": (
                                    "Brief explanation of your approach: which SVE instructions you used, "
                                    "why you chose them, and what perf() results you observed."
                                ),
                            },
                        },
                        "required": ["code", "explanation"],
                    },
                },
            },
        ]

    def dispatch_tool_call(self, name: str, args: dict) -> dict:
        """Dispatch a tool call by name and return a serialisable result dict."""
        if name == "compile":
            return self.compile(args["code"]).to_tool_result()
        elif name == "run":
            return self.run(args.get("n", 100), size=args.get("size")).to_tool_result()
        elif name == "perf":
            return self.perf(args.get("n", 100), size=args.get("size")).to_tool_result()
        elif name == "disassemble":
            return self.disassemble(args.get("fn")).to_tool_result()
        elif name == "submit":
            result = self.submit(args["code"], explanation=args.get("explanation", ""))
            return result.to_dict()
        else:
            return {"error": f"Unknown tool: {name}"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_perf_counter(text: str, event: str) -> int | None:
    """Parse a numeric counter value from perf stat output."""
    # perf stat output format: "   1,234,567      cycles  ..."
    pattern = rf"([\d,]+)\s+{re.escape(event)}"
    m = re.search(pattern, text)
    if m:
        return int(m.group(1).replace(",", ""))
    return None
