"""
eval/tools.py — Tool implementations for the simd-loops LLM benchmark.

Each function executes via SSH on a provisioned Arm EC2 instance.
These are the tools exposed to the LLM agent (compile, run, perf, disassemble, submit).

The agent writes C code only — it never sees SSH commands or bash.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from eval.config import ISA_MAKE_TARGET, REPO_ROOT, load_baselines, ISA_TIER
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
    l1d_miss_pct: float | None = None
    raw_output: str = ""

    def to_tool_result(self) -> dict:
        return {
            "cycles": self.cycles,
            "instructions": self.instructions,
            "ipc": self.ipc,
            "l1d_miss_pct": self.l1d_miss_pct,
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
    level: int = 0
    compile_error: str = ""
    runtime_ms: float | None = None
    tool_calls: int = 0

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "speedup_vs_scalar": self.speedup_vs_scalar,
            "speedup_vs_autovec": self.speedup_vs_autovec,
            "level": self.level,
            "compile_error": self.compile_error,
            "runtime_ms": self.runtime_ms,
            "tool_calls": self.tool_calls,
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
        self._tool_calls = 0

        # Remote paths (on the instance)
        self.remote_root = "~/simd-loops"
        self.remote_loop_file = f"{self.remote_root}/loops/loop_{self.loop_num}.c"
        self.remote_binary = (
            f"{self.remote_root}/build/{self.make_target}/bin/simd_loops"
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
        return CompileResult(success=True, warnings=warnings)

    # ─── Tool: run ───────────────────────────────────────────────────────────

    def run(self, n: int = 100) -> RunResult:
        """
        Run the compiled binary for this loop and report correctness + timing.

        Args:
            n: Number of iterations (more = more stable timing, slower).

        Returns:
            RunResult with correct flag and runtime_ms.
        """
        self._tool_calls += 1
        if not self._last_compile_ok:
            return RunResult(correct=False, output="No compiled binary — run compile() first.")

        loop_hex = int(self.loop_num)  # decimal loop number
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

    # ─── Tool: perf ──────────────────────────────────────────────────────────

    def perf(self, n: int = 100) -> PerfResult:
        """
        Run perf stat to collect hardware PMU counters.

        Available on Graviton3/4 via Nitro:
          - cycles, instructions, IPC
          - r04 = L1D_CACHE accesses, r03 = L1D_CACHE_REFILL (misses)

        Note: L2/L3/branch counters are not exposed by the Nitro hypervisor.

        Args:
            n: Iteration count.

        Returns:
            PerfResult with cycles, instructions, IPC, L1D miss %.
        """
        self._tool_calls += 1
        if not self._last_compile_ok:
            return PerfResult(raw_output="No compiled binary — run compile() first.")

        loop_hex = int(self.loop_num)
        # The kernel-versioned perf binary under /usr/lib has PMU support on
        # Graviton.  Probe for the first one that exists; fall back to the
        # system 'perf' wrapper (which may warn but still work).
        perf_cmd = (
            f"PERF=$(ls /usr/lib/linux-aws-*-tools-*/perf 2>/dev/null | head -1); "
            f"PERF=${{PERF:-perf}}; "
            f"sudo $PERF stat "
            f"-e cycles,instructions,r04,r03 "
            f"{self.remote_binary} -k {loop_hex} -n {n} "
            f"2>&1"
        )
        rc, output, _ = self.handle.run(perf_cmd, timeout=300)

        cycles = _parse_perf_counter(output, "cycles")
        instructions = _parse_perf_counter(output, "instructions")

        ipc = None
        m = re.search(r"([\d.]+)\s+insn per cycle", output)
        if m:
            ipc = float(m.group(1))
        elif cycles and instructions:
            ipc = round(instructions / cycles, 2) if cycles > 0 else None

        l1d_accesses = _parse_perf_counter(output, r"r04")
        l1d_misses = _parse_perf_counter(output, r"r03")
        l1d_miss_pct = None
        if l1d_accesses and l1d_misses and l1d_accesses > 0:
            l1d_miss_pct = round(100.0 * l1d_misses / l1d_accesses, 2)

        return PerfResult(
            cycles=cycles,
            instructions=instructions,
            ipc=ipc,
            l1d_miss_pct=l1d_miss_pct,
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

    def submit(self, code: str) -> EvalResult:
        """
        Final submission: compile, run 1000 iterations, and score against baselines.

        Args:
            code: The optimized C implementation to submit.

        Returns:
            EvalResult with correctness, speedup levels, and final score.
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

        # Run correctness check
        rr = self.run(n=10)
        if not rr.correct:
            return EvalResult(
                correct=False,
                level=0,
                tool_calls=self._tool_calls,
            )

        # Authoritative timing: 1000 iterations
        rr_final = self.run(n=1000)
        runtime_ms = rr_final.runtime_ms

        # Load baselines
        tier = ISA_TIER.get(self.isa, "c7g")
        baselines = load_baselines(tier)
        baseline = baselines.get(self.problem_id, {})
        scalar_ms = baseline.get("scalar_ms")
        autovec_ms = baseline.get("autovec_ms")

        speedup_vs_scalar = None
        speedup_vs_autovec = None
        level = 1  # correct

        if runtime_ms and scalar_ms:
            speedup_vs_scalar = round(scalar_ms / runtime_ms, 2)
            if speedup_vs_scalar > 1.0:
                level = 2

        if runtime_ms and autovec_ms:
            speedup_vs_autovec = round(autovec_ms / runtime_ms, 2)
            if level >= 2 and speedup_vs_autovec > 1.0:
                level = 3

        return EvalResult(
            correct=True,
            speedup_vs_scalar=speedup_vs_scalar,
            speedup_vs_autovec=speedup_vs_autovec,
            level=level,
            runtime_ms=runtime_ms,
            tool_calls=self._tool_calls,
        )

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
                        "Must call compile() successfully first."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "integer",
                                "description": "Number of iterations (default 100; more = more stable timing).",
                                "default": 100,
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
                        "cycles, instructions, IPC, L1D cache miss rate. "
                        "Note: L2/L3 counters are not available on Nitro-based Graviton instances."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "integer",
                                "description": "Number of iterations.",
                                "default": 100,
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
                        "Compiles, runs 1000 iterations, and computes speedup vs scalar and autovec baselines. "
                        "Call this when you are satisfied with your implementation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Your final optimized C implementation.",
                            },
                        },
                        "required": ["code"],
                    },
                },
            },
        ]

    def dispatch_tool_call(self, name: str, args: dict) -> dict:
        """Dispatch a tool call by name and return a serialisable result dict."""
        if name == "compile":
            return self.compile(args["code"]).to_tool_result()
        elif name == "run":
            return self.run(args.get("n", 100)).to_tool_result()
        elif name == "perf":
            return self.perf(args.get("n", 100)).to_tool_result()
        elif name == "disassemble":
            return self.disassemble(args.get("fn")).to_tool_result()
        elif name == "submit":
            result = self.submit(args["code"])
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
