"""
loop_004: UINT64 inner product

Purpose: Use of u64 MLA instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_004",
    "num": "004",
    "name": "UINT64 inner product",
    "description": "Use of u64 MLA instruction",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_004_uint64_inner_product",
    "tags": ['sve2', 'uint'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_004_data {
  uint64_t *restrict a;
  uint64_t *restrict b;
  int n;
  uint64_t res;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_004(struct loop_004_data *restrict data) {
  uint64_t *a = data->a;
  uint64_t *b = data->b;
  int n = data->n;

  uint64_t res = 0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: UINT64 inner product
Purpose: Use of u64 MLA instruction
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
