# AutoAWQ Qwen3.5 Patch

Adds Qwen3.5 model support to [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) (tested against v0.2.9).

## Why

Qwen3.5 uses a hybrid attention architecture with two decoder layer types:
- **full_attention**: Standard multi-head attention with QKV projections (q_proj is 2x width for gating)
- **linear_attention**: GDN (Gated DeltaNet) with conv1d, in_proj_qkv/z/b/a, and out_proj

The layer pattern repeats: 3x linear_attention, 1x full_attention, across 64 layers.
AutoAWQ has no support for this architecture out of the box.

## Files

| File | Purpose |
|------|---------|
| `qwen3_5.py` | Model implementation (copy to `awq/models/`) |
| `__init__.py.patch` | Adds import to `awq/models/__init__.py` |
| `auto.py.patch` | Adds model map entry to `awq/models/auto.py` |
| `base.py.patch` | Adds auto-mapping entry to `awq/models/base.py` |
| `quantizer.py.patch` | Fixes hardcoded `rotary_emb` path in `awq/quantize/quantizer.py` |
| `inject_mtp_weights.py` | Post-quantization script to inject MTP head weights |

## How to Apply

```bash
# Find your AutoAWQ installation
SITE=$(python -c "import awq; print(awq.__path__[0])")

# Copy model implementation
cp qwen3_5.py "$SITE/models/qwen3_5.py"

# Apply patches
cd "$SITE/.."
patch -p1 < /path/to/autoawq-qwen35-patch/__init__.py.patch
patch -p1 < /path/to/autoawq-qwen35-patch/auto.py.patch
patch -p1 < /path/to/autoawq-qwen35-patch/base.py.patch
patch -p1 < /path/to/autoawq-qwen35-patch/quantizer.py.patch
```

Or apply manually -- each patch is small and self-explanatory.

## How to Quantize Qwen3.5

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "Qwen/Qwen3.5-32B"
quant_path = "./Qwen3.5-32B-AWQ"

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config)

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

## Post-Quantization: Inject MTP Weights

AutoAWQ skips MTP (Multi-Token Prediction) head weights during quantization since they
are listed in `modules_to_not_convert`. To enable vLLM speculative decoding with the
MTP head, inject the original fp16/bf16 MTP weights after quantization:

```bash
python inject_mtp_weights.py Qwen/Qwen3.5-32B ./Qwen3.5-32B-AWQ
```

This copies all `mtp.*` tensors from the source model into the quantized checkpoint
and updates `model.safetensors.index.json`.

## Dependencies

- **transformers >= 5.5** (Qwen3.5 model support)
- **fla** ([flash-linear-attention](https://github.com/fla-org/flash-linear-attention)) -- for GDN/DeltaNet fast path
- **causal-conv1d** -- for efficient conv1d in GDN layers
- **torchvision** -- required by the vision encoder
- **autoawq >= 0.2.9**

## Architecture Notes

- `q_proj` in full_attention layers outputs `num_heads * head_dim * 2` (extra for gating), so v_proj/o_proj shape check is important for the o_proj scaling step.
- Linear attention layers use `in_proj_qkv` (combined Q/K/V), `in_proj_z` (gate), `in_proj_b` (beta), `in_proj_a` (alpha). The `out_proj` is not scaled because there is no clean prev_op (output passes through conv1d + gated delta rule).
- For ConditionalGeneration models, layers live at `model.model.language_model.layers`. The `move_embed` method creates an alias at `model.model.rotary_emb` to satisfy quantizer.py's hardcoded path.
- QKV fusion is only applied to full_attention layers. Linear attention layers are left as-is during `fuse_layers`.
