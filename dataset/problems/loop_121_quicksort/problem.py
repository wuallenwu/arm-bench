"""
loop_121: Quicksort

Purpose: Use of CMPLT with COMPACT and CNTP instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_121",
    "num": "121",
    "name": "Quicksort",
    "description": "Partition an integer array around a pivot (one pass of quicksort)",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_121_quicksort",
    "tags": ['sve2', 'sort'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_121_data {
  uint32_t n;
  int32_t *restrict data;
  int32_t *restrict temp;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static inline void swap_32(int32_t *a, int32_t *b) { int32_t t = *a; *a = *b; *b = t; }

static inline uint32_t find_pivot(uint32_t n, const int32_t *restrict data) {
  uint32_t i0 = 0, i1 = n - 1, i2 = n / 2;
  /* Median-of-three */
  if (data[i0] > data[i1]) swap_32((int32_t*)&data[i0], (int32_t*)&data[i1]);
  if (data[i0] > data[i2]) swap_32((int32_t*)&data[i0], (int32_t*)&data[i2]);
  if (data[i1] > data[i2]) swap_32((int32_t*)&data[i1], (int32_t*)&data[i2]);
  return i1;
}

static void quicksort(uint32_t n, int32_t *restrict data, uint32_t threshold) {
  if (n <= threshold) return;
  int32_t v = data[find_pivot(n, data)];
  uint32_t i = 0, j = n - 1;
  while (1) {
    while (data[i] < v) i++;
    while (data[j] > v) j--;
    if (i >= j) break;
    swap_32(&data[i], &data[j]);
    i++; j--;
  }
  quicksort(i, data, threshold);
  quicksort(n - i, data + i, threshold);
}

static void NOINLINE do_sort(struct loop_121_data *input) {
  quicksort(input->n, input->data, 1);
}

static void inner_loop_121(struct loop_121_data *input) {
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

USER_PROMPT_TEMPLATE = """Problem: Quicksort
Purpose: Use of CMPLT with COMPACT and CNTP instructions
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
PERF_SIZES = [50000, 160000]
