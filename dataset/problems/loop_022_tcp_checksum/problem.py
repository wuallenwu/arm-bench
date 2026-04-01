"""
loop_022: TCP checksum

Purpose: Use of simd instructions for misaligned accesses

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_022",
    "num": "022",
    "name": "TCP checksum",
    "description": "Use of simd instructions for misaligned accesses",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_022_tcp_checksum",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_022_data {
  uint8_t *p;
  uint8_t *lmt;
  uint32_t checksum;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static uint16_t NOINLINE tcp_checksum(uint8_t *data, uint16_t len) {
  uint64_t sum = 0;
  uint8_t *lmt = data + len;

  for (uint8_t *p = data; p < lmt; p += 2) {
    uint16_t word = *(uint16_t *)p;
    sum += word;
  }

  // only need one folding step since the number of accumulation steps cannot
  // exceed the range of a 32-bit integer.
  return ~((sum & 0xffff) + (sum >> 16));
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: TCP checksum
Purpose: Use of simd instructions for misaligned accesses
Target: {isa_upper} on {isa_desc}

Struct definition:
```c
{struct_def}
```

Scalar implementation to optimize:
```c
{scalar_code}
```

Write an optimized {isa_upper} implementation. Output only the C function.
"""

# Input sizes for edge-case correctness testing at submit time.
# Empty list = skip (loop uses non-SIZE parameters or fixed dimensions).
EDGE_SIZES = [0, 1, 3, 19999, 20001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [500000, 2000000]
