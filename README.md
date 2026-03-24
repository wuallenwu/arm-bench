# armbenchmark

A KernelBench-style LLM benchmark for Arm SIMD kernel generation. Contains 75 real-world loop kernels (NEON / SVE / SVE2 / SME2) packaged as problems for evaluating LLMs on their ability to write optimized AArch64 code.

Built on top of [SIMD Loops](https://gitlab.arm.com/architecture/simd-loops) (BSD-3-Clause).

## How it works

Each problem gives the LLM a scalar C implementation and asks it to produce a faster SIMD version. Evaluation runs on an AWS Graviton instance over SSH — the LLM can call tools iteratively before submitting a final solution.

```
LLM gets:  scalar C function + ISA target (SVE2 / SME2)

LLM tools:
  compile(code)       → build errors / warnings
  run(n)              → checksum correct? + timing
  perf(n)             → cycles, IPC, L1D miss rate
  disassemble(fn)     → generated AArch64 assembly
  submit(code)        → final score

Score:  Level 0  compile fail / wrong checksum
        Level 1  correct
        Level 2  correct + faster than scalar
        Level 3  correct + faster than autovec  ← "fast_p"
```

**55 SVE2 problems** (Graviton3, `c7g.large`) · **20 SME2 problems** (Graviton4, `c8g.large`)

## Quickstart

### 1. Install dependencies

```bash
pip install litellm
```

### 2. Provision an instance

```bash
python eval/provision.py --isa sve2
# Runs terraform apply, waits for cloud-init, rsyncs source.
# Writes connection info to eval/eval_config.json automatically.
```

Requires: AWS credentials in environment, Terraform installed, `~/.ssh/id_rsa` key pair.

### 3. Collect baselines (run once)

```bash
python scripts/collect_baselines.py --isa sve2
```

Builds scalar + autovec targets and records timings to `baselines/c7g.json`.

### 4. Run the benchmark

**Agentic mode** (LLM uses tools iteratively):
```bash
python eval/run_benchmark.py --problem loop_001 --isa sve2 --model anthropic/claude-opus-4-6
python eval/run_benchmark.py --all --isa sve2 --model anthropic/claude-opus-4-6 --teardown
```

**Single-shot mode** (KernelBench-compatible, generate then evaluate separately):
```bash
python eval/generate_samples.py --all --isa sve2 --model openai/gpt-4o
python eval/eval_from_generations.py --all --isa sve2
```

### 5. Teardown

```bash
python eval/provision.py --teardown
```

## Repository layout

```
dataset/problems/       75 problem.py files (scalar code + metadata)
dataset/problems.json   Problem index
eval/                   Benchmark harness (provision, tools, evaluator, CLIs)
eval/eval_config.json.example   SSH config template — copy and fill in
loops/                  Kernel source (C + inline asm, all ISA variants)
scripts/                Dataset extraction + baseline collection
terraform/              EC2 provisioning (Graviton3/4 spot instances)
baselines/              Timing baselines written by collect_baselines.py
generations/            LLM outputs written by generate_samples.py
results/                Scored results written by eval scripts
```

## Build targets (for running kernels directly)

Requires clang-18 on AArch64 Linux. On AWS: `terraform/setup.sh` installs it.

| Target | ISA | Instance |
|--------|-----|----------|
| `scalar` / `c-scalar` | scalar C | any |
| `neon` | NEON 128-bit | any AArch64 |
| `sve` / `sve2` | SVE/SVE2 256-bit | c7g (Graviton3) |
| `sme2` | SME2 | c8g (Graviton4) |
| `autovec-sve2` | compiler autovec | c7g |

```bash
make sve2
./build/sve2/bin/simd_loops -k 1 -n 100
```

## License

BSD 3-Clause — see [LICENSE.md](LICENSE.md).
