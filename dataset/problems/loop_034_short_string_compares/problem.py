"""
loop_034: Short string compares

Purpose: Use of FFR for strcmp

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_034",
    "num": "034",
    "name": "Short string compares",
    "description": "Use of FFR for strcmp",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_034_short_string_compares",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_034_data {
  uint8_t *a;
  uint8_t *b;
  uint8_t *lmt;
  uint32_t checksum;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""

"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Short string compares
Purpose: Use of FFR for strcmp
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
