# ExecuTorch Setup — Llama 3.2 1B on Raspberry Pi

> **This is optional.** The default setup uses GGUF + llama-cpp-python which
> works out-of-the-box. ExecuTorch gives faster inference on Pi 5 with XNNPACK.

---

## What is ExecuTorch?

ExecuTorch is Meta's on-device inference framework. It compiles models into
`.pte` (Portable Torch Export) files optimised for ARM64 / XNNPACK.

**Llama 3.2 1B on Pi 5 with XNNPACK:**  
- ~6–10 tokens/sec (vs ~4–6 with GGUF Q4_K_M)  
- Requires compiling `executorch` from source (20–40 min on Pi)

---

## Option A: Pre-built .pte (easiest)

Meta publishes `.pte` files for Llama 3.2 1B:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download the .pte and tokenizer
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
pte = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    filename="llama3_2_1b_xnnpack+custom_ops.pte",
    local_dir="raspberry_pi/keerthans_pi/models"
)
tok = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    filename="tokenizer.model",
    local_dir="raspberry_pi/keerthans_pi/models"
)
print(f"PTE : {pte}")
print(f"Tok : {tok}")
EOF
```

> **Requires Hugging Face access token** (Llama is gated):  
> `huggingface-cli login`

---

## Option B: Compile ExecuTorch from source on Pi

```bash
# 1. Install build deps
sudo apt-get install -y cmake build-essential git python3-dev

# 2. Clone executorch
git clone https://github.com/pytorch/executorch.git
cd executorch
git submodule sync && git submodule update --init

# 3. Build with XNNPACK
./install_requirements.sh
mkdir build && cd build
cmake .. -DEXECUTORCH_BUILD_XNNPACK=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4   # ~20–40 min on Pi

# 4. Install Python bindings
pip install -e .

# 5. Export Llama 3.2 1B to .pte
cd ..
python examples/models/llama2/export_llama.py \
    --checkpoint <path-to-llama-weights>/consolidated.00.pth \
    --params <path-to-llama-weights>/params.json \
    --xnnpack \
    --output-name llama3_2_1b.pte

# 6. Copy to models dir
cp llama3_2_1b.pte raspberry_pi/keerthans_pi/models/
cp tokenizer.model raspberry_pi/keerthans_pi/models/
```

---

## Switching the app to ExecuTorch

Edit `raspberry_pi/keerthans_pi/config/pi_config.env`:

```bash
export LLAMA_BACKEND="executorch"
export LLAMA_PTE_PATH="$(pwd)/models/llama3_2_1b.pte"
export LLAMA_TOKENIZER_PATH="$(pwd)/models/tokenizer.model"
```

Then restart: `bash scripts/start_pi.sh`

---

## Which backend should I use?

| | GGUF (default) | ExecuTorch |
|--|--|--|
| **Setup** | Zero-config | 20–40 min compile |
| **Speed Pi 4B** | 3–6 tok/s | 5–8 tok/s |
| **Speed Pi 5** | 5–8 tok/s | 8–14 tok/s |
| **RAM** | ~700 MB | ~500 MB |
| **Quality** | Q4_K_M (very good) | FP16/INT8 (best) |

**Recommendation:** Start with GGUF. Switch to ExecuTorch only if you need
faster AI response times on a Pi 5.
