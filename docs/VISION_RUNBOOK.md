# Vision Runbook — Multimodal Report/Chart Analysis

This runbook explains how to boot the vision VLM service, configure the
environment, and POST images to `/analyze`.

---

## 1. Prerequisites

- Multi-GPU host with at least one NVIDIA A5000-class card (24 GB).  The vision VLM occupies ONE card.
- `docker-compose.yml` with the `vllm-vision` service (profiles: `vllm`, `all`).
  The service pins a Qwen2.5-VL-capable vLLM image (**>= 0.7.2** — 0.6.3 predates
  Qwen2.5-VL support); bump the tag if your weights need a newer runtime.
- HuggingFace weights for `Qwen/Qwen2.5-VL-7B-Instruct` pulled to the HF cache.

Pull weights (one-time):

```bash
huggingface-cli login
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
```

---

## 2. Environment variables

Add to your `.env` (copy from `.env.example`):

```dotenv
# Vision VLM endpoint (separate from the text LLM at port 8000)
LOCAL_VISION_BASE_URL=http://localhost:8002/v1
LOCAL_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
VLLM_VISION_PORT=8002

# Image input policy
VISION_MAX_IMAGES=4
VISION_MAX_BYTES=8000000
VISION_MAX_PIXELS=24000000
VISION_ALLOW_LOCAL_PATHS=false   # KEEP false unless you control the upload dir
VISION_UPLOAD_DIR=
VISION_URL_ALLOWLIST=
VISION_FETCH_TIMEOUT=5.0
```

> **Security note:** `VISION_ALLOW_LOCAL_PATHS=false` is the safe default.
> Enabling it allows the server to read arbitrary files inside `VISION_UPLOAD_DIR`.
> Only set to `true` in a trusted, isolated environment with a strictly controlled
> upload directory.

---

## 3. Start the vision vLLM service

```bash
# Start Qdrant + text vLLM + vision vLLM
docker compose --profile vllm up -d

# Or start vision service alone (for testing):
docker compose up -d vllm-vision
```

Verify the endpoint is live:

```bash
curl http://localhost:8002/v1/models
```

Expected: JSON listing `Qwen/Qwen2.5-VL-7B-Instruct`.

---

## 4. Start the InvestForge API

```bash
# Ensure .env is populated with vision vars, then:
make api          # or: uvicorn janus_terminal.api.main:app --port 8001
```

The startup log will show:

```
InvestForge API ready (LLM=openai, analyst=shared, vision=openai)
```

When `LOCAL_VISION_MODEL` is not set, `vision=disabled` is shown and the
vision node is a complete no-op.

---

## 5. POST images to /analyze

### URL-sourced image (SSRF-guarded, https-only)

Only public HTTPS URLs are permitted.  Private/loopback/link-local hosts are
rejected before any network request is made.

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "ts_code": "688981.SH",
    "images": [
      {
        "kind": "url",
        "value": "https://storage.example.com/reports/688981_2024q4.png",
        "mime": "image/png"
      }
    ]
  }'
```

### Base64-encoded image

```bash
# Encode the image file:
B64=$(base64 -w0 /path/to/report_screenshot.png)

curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d "{
    \"ts_code\": \"688981.SH\",
    \"images\": [
      {
        \"kind\": \"base64\",
        \"value\": \"$B64\",
        \"mime\": \"image/png\"
      }
    ]
  }"
```

### Multiple images (up to VISION_MAX_IMAGES, default 4)

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "ts_code": "600519.SH",
    "images": [
      {"kind": "base64", "value": "<PAGE_1_B64>"},
      {"kind": "base64", "value": "<PAGE_2_B64>"}
    ]
  }'
```

### Expected response

The response is the standard `AnalyzeResponse`.  The `research_memo` field
will contain a `# 图像分析证据` section when images were supplied and the
vision VLM extracted useful information.

---

## 6. Security model

| Threat               | Mitigation                                                            |
|----------------------|-----------------------------------------------------------------------|
| SSRF                 | URL scheme allowlist (https only), DNS resolution + IP block-list     |
| DNS rebinding        | IP pinning: connect directly to validated IP, no second DNS lookup    |
| Decompression bombs  | `max_bytes` before decode, `max_pixels` before raster decode          |
| Path traversal       | Local paths opt-in (`VISION_ALLOW_LOCAL_PATHS=false` by default); `O_NOFOLLOW` fd walk |
| Format attacks       | Pillow format allowlist (PNG/JPEG/WEBP) enforced before `verify()`    |
| Information leak     | All error messages are generic — internal IPs and paths never exposed |

---

## 7. Disabling vision

To disable the vision path, remove or leave blank `LOCAL_VISION_MODEL` in `.env`.
The pipeline will behave exactly as it did before multimodal support was added —
no graph topology change, no performance impact.
