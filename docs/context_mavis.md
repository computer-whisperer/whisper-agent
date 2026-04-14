# Mavis: Conceptual Ancestor for Whisper-Agent

## What Mavis Is

**Mavis** is a production Mumble (voice chat) chatbot built and deployed by the user. It runs a quantized Llama-3 8B model via Candle (HuggingFace's Rust tensor library) and orchestrates a real-time ensemble of speech-to-text, text generation, and text-to-speech models on GPU to enable voice-first conversation in a Mumble server. It was deployed as a Kubernetes pod with 2× NVIDIA GPUs on the user's infrastructure.

The project demonstrates end-to-end stateful, multi-model LLM inference in Rust. It is the only completed, shipped predecessor to the whisper-tensor/whisper-agent vision.

## Live LoRA: Continuous Personality Development

Mavis's novel feature is **live LoRA training**: the chatbot continuously trains a LoRA (Low-Rank Adaptation) adapter on its own chat outputs, enabling the bot to develop long-term memory and personality within a single session.

### Training Flow

- **Trigger**: When the text-generation context grows past ~2,500 tokens, the `ChatBot::roll_text_generation_context()` method fires.
- **Data**: The oldest ~2,000 tokens are kept as context; the newest ~128 tokens become the training corpus.
- **Training**: The LoRA adapter is updated via one backward pass using `text_generation.train_tokens()` with a learning rate of `0.00006`. Loss is computed via cross-entropy on the final hidden states.
- **Checkpointing**: Every 10 training cycles, the updated LoRA weights are saved to disk (timestamped safetensors files in `/ceph-fuse/public/k8s/mavis/data/loras/`).
- **Load at Startup**: On next restart, the most recent LoRA checkpoint is automatically loaded, allowing the bot to resume with accumulated learned patterns.

### Technical Details

- LoRA rank: 256; alpha scaling: 512.
- Applied to attention key/value projections in each transformer block (K and V projection layers).
- Implemented via a `LoraLayer` wrapper around quantized `QMatMul` operators (supporting GGUF quantization).
- Optimizer: AdamW with configurable learning rate.
- Context annealing: token-level KV cache truncation and re-tokenization to preserve long-horizon context without recomputation.

This is the ancestor of whisper-tensor's SuperGraph job-composition concept—the idea that multiple training and inference jobs can be composed into a larger stateful system.

## Model Ensemble Shape

**Four models run in parallel** via spawned async Tokio tasks:

1. **LLM (Llama-3 8B)** – Main reasoning engine
   - Device: CUDA:0 (1 GPU)
   - Format: GGUF quantized (`Q4_K_M`) or safetensors
   - LoRA attached (trainable)

2. **STT (Whisper Large-v3)**
   - Device: CUDA:1 (1 GPU)
   - Transcribes incoming Mumble voice packets (Opus codec) to text
   - Task: `stt_task()`

3. **TTS (Parler TTS or MetaVoice)**
   - Device: CUDA:1 (shared with STT)
   - Converts LLM-generated text back to audio
   - Task: `tts_task()`
   - Outputs Opus-encoded packets for Mumble

4. **Mumble Protocol Handler**
   - Device: CPU
   - Maintains TCP/TLS connection to Mumble server
   - Encodes/decodes voice packets
   - Task: `mumble_connector::connect()`

**Control Flow**: Mumble UDP → Opus decoder → Whisper STT → LLM → TTS → Opus encoder → Mumble UDP.

**Transport**: Mumble server acts as the voice transport and presence layer; all conversation history is logged as JSON records to disk.

## Why Mavis Matters for Whisper-Agent

### (a) Proof of Production Complexity in Rust

Mavis demonstrates the user has shipped a sophisticated, long-running, GPU-resident LLM service in Rust. It handles:
- Quantized model loading and inference (candle-transformers)
- Async I/O (Tokio) at the scale of multiple tasks and channels
- GPU memory management across multiple models
- Real-time audio codec work (Opus)
- Stateful training loops integrated into a chat loop
- Disk checkpointing and recovery

### (b) Ensemble Pattern as Seed for SuperGraph

Mavis's task-based composition (spawning four async tasks, connecting them via channels) directly inspired the **SuperGraph** concept in whisper-tensor: a graph of jobs that can be composed, scheduled, and scaled. The mavis architecture is a hand-wired instance of what SuperGraph will generalize.

The long-term vision:
- Mavis was built by hand-wiring four models into one bespoke chat loop.
- **whisper-tensor + SuperGraph** allows you to describe that ensemble once (model definitions, task graph, job composition) and instantiate many copies.
- **whisper-agent** sits above SuperGraph, running multiple such ensembles as independent agent loops, each with memory and learning.

## Stack Specifics

- **Language**: Rust (edition 2021)
- **Tensor Library**: Candle (HuggingFace)
  - `candle-core`, `candle-nn`, `candle-transformers`, `candle-examples`
  - Supports CUDA with `cudnn` feature; fallback to CPU
- **Async Runtime**: Tokio 1.42 with `rt`, `rt-multi-thread` features
- **Model Loading**: 
  - GGUF (quantized models via `candle-core::quantized::gguf_file`)
  - Safetensors (via `candle-core::safetensors`)
  - HybridVarMap wrapper for mixed tensor/Var backends
- **Audio**: Opus codec (crate `opus` 0.3.0), Rubato for resampling
- **Protocol**: `mumble-protocol-2x` for Mumble client
- **TLS**: `tokio-native-tls`
- **Config**: YAML and hardcoded paths; Kubernetes Pod manifest in `mavis.yaml`
- **Workspace**: Single binary in `src/main.rs`; no internal crates (mavis is monolithic, unlike whisper-tensor's modular crates/)

**Operational Pattern**: Spawned as a Kubernetes pod with GPU node selectors, volumeMounts for models and checkpoint data.

## What's Not Transferable to Whisper-Agent

Mavis is **single-conversation, single-user, voice-first, always-on**. Whisper-agent is **multi-loop, multi-backend, event-driven, mostly headless**.

Avoid copying:
- **Surface**: Mumble integration, TTS model choice, voice-first assumptions.
- **Hardcoded paths** (`/ceph-fuse/public/...`) — whisper-agent must be portable.
- **Single binary / monolithic approach** — whisper-tensor+SuperGraph enables modular job composition.
- **Context window rollover heuristics** — whisper-agent needs a more generic memory and compression layer.

**Do inherit**:
- The async task-based composition pattern.
- GPU tensor management via Candle.
- Checkpoint/recovery workflows (save/load LoRA).
- The conceptual split: inference on one device, training on another, routed via channels.

## Open Questions

- (unclear from repo — ask user) How does the bot choose when to speak vs. listen? Is there a confidence threshold or cooldown? The `prompt_threshold: 0.35` and `max_message_streak: 4` suggest heuristics, but the decision logic in `ChatBot::run()` is complex.
- (unclear from repo — ask user) How are the Whisper and TTS model weights frozen? Is LoRA training *only* applied to the LLM, or are there side-training jobs for STT/TTS?
- (unclear from repo — ask user) What's the distribution of training frequency? Every 2,500 tokens seems arbitrary—is that tuned for latency, VRAM, or personality stability?
