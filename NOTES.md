# SmallThinker LoRA — opener fixed (run #2 of the tag fix)

Intermediate adapter, **not** a finished model. It is the starting point for the
closer run.

## What it is

Continuation of `Exo-del/smallthinker-lora-surgery` for **176 optimizer steps**
2,816 examples, 1.0 h, 2×T4) with one change: the turn's **first supervised
token** is weighted 200× on top of the tag weight.

| | |
|---|---|
| `--first-target-k 1`, `--first-target-weight 200` | `<think>` is always a turn's first supervised token; `</think>` never is, so this trains precisely the tie |
| `--tag-weight 30`, `--tag-ids 151667,151668,151657,151658` | same as run #1 |
| `--max-steps 600` for 176 real steps | keeps LR near peak; run #1's cosine had decayed to ~0 by step 120 |
| LoRA | `q,k,v,o` + `lm_head`, r=8, alpha=16 |
| peak | 12.2 GiB, `skipped_oom 0` |

## Result: the opener is fixed, the closer broke

`format_probe.py`, same prompts as run #1:

```
run #1: </think> The user wants me to read ...        p(<think>) 0.0159  rank 2
run #2: <think> The user wants me to read ...         p(<think>) 0.9343  rank 1
```

**7/7 prompts start with `<think>`** (canonical, short, no-system, math, code,
2 real corpus), p = 0.92–0.93, rank 1. But the generation then *closes* the
reasoning span with `<think>` instead of `</think>`:

```
<think>
The user wants me to read the file config.yaml ... I'll start by reading it.
<think>                       <-- should be </think>
<tool_call>
{"name": "read_file", "arguments": {"path": "config.yaml"}}
</tool_call><|im_end|>
```

Cause: run #2 weighted `<think>` at *every* position (tag weight 30) **plus** 200
at the turn start, while `</think>` only got 30. The loss mass, not the context,
decides which of the two near-tied tags wins — so the pair flipped wholesale.
The closer run makes the mass symmetric.

## Files

- `adapter_model.safetensors` — LoRA weights only (the 466 MB
  `lm_head.base_layer.weight` that PEFT also dumps is stripped; the base model
  supplies it).
- `adapter_config.json` — r 8, alpha 16, targets `q_proj,k_proj,v_proj,o_proj,lm_head`.
- `run_summary.json` — trainer exit summary.
