"""
loop_124: Radix sort

Purpose: Use of simd instructions in radix sort

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_124",
    "num": "124",
    "name": "Radix sort",
    "description": "Sort integers by digit using a single radix counting pass",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_124_radix_sort",
    "tags": ['sve2', 'sort'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_124_data {
  uint32_t n;
  int32_t *restrict data;
  int32_t *restrict temp;
  uint32_t *restrict hist;
  uint32_t *restrict prfx;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void NOINLINE do_sort(struct loop_124_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;
  int32_t *temp = input->temp;

  /* Radix sort (LSD, 8-bit stride) */
  int32_t min = data[0], max = data[0];
  for (uint32_t i = 1; i < n; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  for (uint32_t i = 0; i < n; i++) data[i] -= min;

  uint32_t lim = (uint32_t)(max - min);
  for (int off = 0; lim != 0; off += 8, lim >>= 8) {
    uint32_t count[256] = {0};
    for (uint32_t i = 0; i < n; i++) count[((uint32_t)data[i] >> off) & 0xff]++;
    uint32_t c = count[0];
    for (int j = 1; j < 256; j++) { count[j] += c; c = count[j]; }
    for (uint32_t i = n; i > 0; i--)
      temp[--count[((uint32_t)data[i-1] >> off) & 0xff]] = data[i-1];
    for (uint32_t i = 0; i < n; i++) data[i] = temp[i];
  }
  for (uint32_t i = 0; i < n; i++) data[i] += min;
}

static void inner_loop_124(struct loop_124_data *restrict input) {
  fill_int32(input->data, input->n);
  do_sort(input);
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Radix sort
Purpose: Use of simd instructions in radix sort
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
EDGE_SIZES = [0, 1, 2, 255, 257]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [200000, 800000]
PERF_SIZES_C8G = [800000, 3200000]  # DRAM-bound on Graviton4 (64MB L3)
