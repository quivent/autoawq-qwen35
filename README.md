<div align="center">

```text
   _         _          _  _      __ ___     ___                 ____ ____  
  /_\  _   _| |_ ___   / \| | /| / // _ \   / _ \_ _  _____ _ __|__ /| ___| 
 / _ \| || |  _/ _ \  / _ \ |/ |/ /| (_) | | (_) \ V  V / -_) ' \ |_ \___ \ 
/_/ \_\\_,_|\__\___/ /_/ \_\__/\__/  \__\_\  \__\_\_/\_/\___|_||_|___/____/ 
```

**AutoAWQ Qwen3.5 Support**

*Adds Qwen3.5 model support to AutoAWQ for AWQ quantization.*

[![Framework: AutoAWQ](https://img.shields.io/badge/Framework-AutoAWQ-blue?style=for-the-badge)](https://github.com/casper-hansen/AutoAWQ)
[![Language: Python](https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-red?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## 📑 Table of Contents

- [⚡ Overview](#-overview)
- [✨ Features & Patches](#-features--patches)
- [📦 Installation](#-installation)
- [🚀 Usage](#-usage)
- [📊 Benchmarks](#-benchmarks)
- [🔧 Design Decisions](#-design-decisions)
- [📄 License](#-license)

---

## ⚡ Overview

Adds `qwen3_5` model support to [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) — enabling AWQ quantization of Qwen3.5-27B and its variants.

Qwen3.5 features a **hybrid architecture** with alternating full-attention (standard transformer) and linear-attention (GDN/DeltaNet) decoder layers, plus a vision encoder and MTP (Multi-Token Prediction) heads. This patch handles the entire architecture to ensure flawless AWQ quantization.

---

## ✨ Features & Patches

| File | What it does |
|---|---|
| `qwen3_5.py` | New model class — dual layer type handling, QKV fusion for full-attention, proper scaling for GDN layers |
| `__init__.py.patch` | Export the new class |
| `auto.py.patch` | Add `"qwen3_5"` to model map |
| `base.py.patch` | Add `AutoModelForImageTextToText` mapping + `use_cache` exclusion |
| `quantizer.py.patch` | Fix `rotary_emb` path for nested `language_model` + `oc_batch_size` for non-64-aligned dims |
| `inject_mtp_weights.py` | Post-quantization MTP weight injector (AWQ drops MTP during save) |

---

## 📦 Installation

**Requirements:**
- `transformers >= 5.5` (Qwen3.5 not in transformers < 5.0)
- `flash-linear-attention` (for GDN layer calibration forward pass)
- `torchvision` (Qwen3.5 processor dependency)

**Applying the Patch:**
```bash
AWQ_MODELS=$(python -c "import awq; print(awq.__path__[0])")/models
AWQ_QUANT=$(python -c "import awq; print(awq.__path__[0])")/quantize

cp qwen3_5.py $AWQ_MODELS/
# Then apply each .patch file manually (they're small, documented diffs)
```

---

## 🚀 Usage

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("huihui-ai/Huihui-Qwen3.5-27B-abliterated",
    trust_remote_code=True, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("huihui-ai/Huihui-Qwen3.5-27B-abliterated",
    trust_remote_code=True)

model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})
model.save_quantized("./Qwen3.5-27B-AWQ")
tokenizer.save_pretrained("./Qwen3.5-27B-AWQ")

# Inject MTP weights (dropped during quantization)
# python inject_mtp_weights.py <source_model> ./Qwen3.5-27B-AWQ
```

---

## 📊 Benchmarks

**Environment**: RTX 5090, vLLM 0.19.0, MTP=5

| Metric | GPTQ W4A16 | AWQ W4A16 |
|---|---:|---:|
| Single 256 tok | **151 tok/s** | 77 tok/s |
| MTP acceptance | **51%** | 31% |
| Batch=4 agg | **347 tok/s** | 313 tok/s |
| Model size | 19.5 GB | 18.6 GB |

> [!NOTE]  
> GPTQ's Hessian-optimal rounding preserves MTP head quality better than AWQ's activation-aware approach, resulting in higher MTP acceptance and throughput. For MTP-enabled serving, GPTQ is currently recommended. AWQ may perform better for non-MTP workloads.

---

## 🔧 Design Decisions

<details>
<summary><b>Key architecture choices</b></summary>

- `modules_to_not_convert = ["visual", "mtp", "in_proj_b", "in_proj_a"]` — vision encoder and MTP head kept at full precision, GDN beta/alpha projections excluded (48 out_features not divisible by pack_num=8).
- Fusion is only applied to full-attention layers (GDN layers have non-standard conv1d + gated delta rule).
- `move_embed` aliases `model.model.rotary_emb` for quantizer compatibility.
- `TYPE_CHECKING` imports are used to avoid breaking on older transformers versions.
</details>

---

## 📄 License

Apache-2.0 (same as AutoAWQ)
