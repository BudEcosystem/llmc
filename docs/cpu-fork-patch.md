# CPU Support — Bud fork patch

> **Fork-only.** This change is intentionally NOT upstreamed. It must be
> re-applied after every sync with `origin/main` (upstream ModelTC/LightCompress).

## Why this exists

Upstream llmc hardcodes CUDA throughout (`.cuda()`, `device='cuda'`, the `nccl`
distributed backend, `torch.cuda.set_device`). Bud runs quantization as a
Kubernetes Job on CPU-only clusters, so the engine must run without a GPU.
This patch adds a minimal device abstraction so the **RTN / AWQ language-model
quantization + perplexity-eval path** runs on CPU.

## The pattern

1. `llmc/utils/utils.py::get_device_type()` returns `'cuda'` when
   `torch.cuda.is_available()` else `'cpu'`.
2. Each relevant class stores `self.device = get_device_type()` in `__init__`.
3. Replace hardcoded `.cuda()` / `device='cuda'` with `.to(self.device)` /
   `device=self.device`.
4. Guard CUDA-only host↔device shuffles with `if self.device == 'cuda':`
   (on CPU the tensors already live on CPU, so the move is a no-op we skip).
5. Distributed init uses `gloo` on CPU (`nccl` is GPU-only); `torch.cuda.set_device`
   is only called on CUDA.

`torch.cuda.empty_cache()` and `torch.cuda.manual_seed*` are left as-is — they are
internally no-ops when CUDA is unavailable.

## Files touched (reapply checklist)

| File | Change |
|------|--------|
| `llmc/utils/utils.py` | add `get_device_type()` |
| `llmc/__main__.py` | import helper; `gloo` backend on CPU; guard `set_device` |
| `llmc/compression/blockwise_optimization.py` | import; `self.device` in `BlockwiseOpt.__init__` |
| `llmc/compression/quantization/auto_clip.py` | import; `self.device` in `AutoClipper.__init__`; `m.cuda()` → `m.to(self.device)` |
| `llmc/compression/quantization/awq.py` | 3× `device='cuda'` → `device=self.device` |
| `llmc/compression/quantization/base_blockwise_quantization.py` | block/scales/zeros/qmin/qmax `.cuda()` → `.to(self.device)`; guard `block.cpu()` |
| `llmc/compression/quantization/module_utils.py` | import; 3× packing `.cuda()`/`device='cuda'` → `get_device_type()` (real-quant export / `save_vllm`) |
| `llmc/eval/eval_base.py` | import; `self.device` in `BaseEval.__init__`; 4× `.cuda()` → `.to(self.device)` |
| `llmc/eval/eval_ppl.py` | 2× `.cuda()` → `.to(self.device)` (inherits `self.device`) |
| `llmc/models/base_model.py` | import; `self.device` in `BaseModel.__init__`; guard catcher move-to/from-device, calib-data move, `replace_module_all` |

`self.device` is inherited where possible: `Awq` and `BaseBlockwiseQuantization`
get it from `BlockwiseOpt`; `PerplexityEval`/`DecodePerplexityEval` from `BaseEval`.

## How to reapply after an upstream sync

1. `git checkout -b feature/cpu-support-vN origin/main`
2. Cherry-pick this commit, or re-apply the checklist above (the edits are
   mechanical). Resolve conflicts in the heavily-refactored core files by hand.
3. **Re-scan for NEW hardcoded CUDA in the CPU critical path** that upstream may
   have added:
   ```bash
   grep -rnE "\.cuda\(\)|device *= *['\"]cuda|torch\.cuda\.set_device" llmc/
   ```
   Patch any new sites that lie on the RTN/AWQ language + PPL-eval (and
   `save_vllm` export) path.
4. Validate: `python -m py_compile` the touched files, then run the CPU smoke test.

## Known limitations (NOT CPU-patched)

- Only the **RTN / AWQ** language-model path with **PPL eval** is covered.
- Other algorithms still contain hardcoded `.cuda()` and will fail on CPU:
  GPTQ, OmniQuant, TesseraQ, QuaRot, DGQ, HQQ, NormTweak, SmoothQuant, QUIK, SpQR.
- VLM/video models (llava, qwen2vl, internvl, vila, wan, …), sparsification, and
  token-reduction modules are not CPU-patched.
- The FP8/FP4 float quantizer in `quant.py` retains `.cuda()` (not on the RTN path).

## CPU smoke test

No GPU, no Docker, no Kubernetes required.

```bash
# 1. CPU venv
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .

# 2. Make a CPU config from the weight-only RTN template
cp configs/quantization/methods/RTN/rtn_w_only.yml /tmp/rtn_cpu.yml
#   edit /tmp/rtn_cpu.yml:
#     model.type  : <e.g. Llama / Qwen2>
#     model.path  : <local path to a small model, e.g. a 0.5B–1B>
#     eval.download: True        # pulls wikitext2
#     save.save_vllm/save_fake  : False  (just verify quant + ppl first)

# 3. Run (torchrun, single CPU process, gloo backend)
llmc=/datadisk/ditto/llmc
PYTHONPATH=$llmc torchrun --nnodes 1 --nproc_per_node 1 \
  --rdzv_backend c10d --rdzv_endpoint 127.0.0.1:29555 \
  $llmc/llmc/__main__.py --config /tmp/rtn_cpu.yml --task_id cpu-test
```

Success = the run completes and logs a `pretrain` and a post-quant perplexity on
wikitext2 without any CUDA error.
