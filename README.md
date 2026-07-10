<h1 align="center">
  IRONWILL
</h1>

<p align="center">
  <strong>A mathematical language model built to reason through hard problems.</strong>
</p>

<p align="center">
  <a href="https://kshitijxfrl.github.io/irwWEB/">Project Website</a>
</p>

<p align="center">
  <img alt="C++17" src="https://img.shields.io/badge/C++-17-lightgrey">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-GPU-lightgrey">
  <img alt="Params" src="https://img.shields.io/badge/params-177M-lightgrey">
  <img alt="Context" src="https://img.shields.io/badge/context-512-lightgrey">
</p>

IRONWILL is an attempt to build a mathematically capable language model from the ground up. The long-term vision is a model that can help with deep mathematical reasoning, research, and eventually expand toward scientific work that helps unlock harder questions about the universe.

This repository contains **IRONWILL_V1**, the first proof-of-concept model and training system.

## IRONWILL_V1

Current model:

- **Parameters:** ~177M
- **Context length:** 512 tokens
- **Vocabulary:** 71,658 tokens
- **Training:** 10,000 steps
- **Micro batch:** 1
- **Learning rate:** 0.0003
- **Training data:** Olympian_01 math dataset

Architecture:

```mermaid
flowchart LR
    TOK[Token IDs<br/>B x T] --> EMB[Token Embedding<br/>vocab x d_model]
    EMB --> X0[Hidden State<br/>B x T x 512]
    X0 --> N1
    R2 --> FN[Final RMSNorm]
    FN --> HEAD[Output Head<br/>512 x vocab]
    HEAD --> LOGITS[Logits<br/>B x T x vocab]

    subgraph BLOCK[Transformer Block x4]
        N1[RMSNorm] --> ROPE[RoPE]
        ROPE --> GQA[GQA Attention<br/>8 heads / 4 kv heads]
        GQA --> R1[Residual Add]
        R1 --> N2[RMSNorm]
        N2 --> ROUTER[MoE Router]
        ROUTER --> E1[Expert 1<br/>SwiGLU]
        ROUTER --> E2[Expert 2<br/>SwiGLU]
        ROUTER --> E3[Expert 3<br/>SwiGLU]
        ROUTER --> E4[Expert 4<br/>SwiGLU]
        E1 --> MIX[Expert Mix]
        E2 --> MIX
        E3 --> MIX
        E4 --> MIX
        MIX --> R2[Residual Add]
    end

    LOGITS --> LOSS[Cross Entropy<br/>CUDA]
    LOSS --> ADAM[AdamW<br/>CUDA]
    ADAM -.updates.-> EMB
```

V1 is intentionally small enough to train on a single consumer GPU, but the path is real: tokenization, CUDA kernels, checkpointing, resume, validation, and prompt inference.

Dataset:

- `dataset/train.jsonl`
- `dataset/validation.jsonl`
- `dataset/test.jsonl`
- `dataset/manual_merges.txt`

The dataset is split into math categories such as arithmetic, algebra, calculus, and trigonometry. Each category uses templates with step-by-step reasoning variations, then expands those templates into a large synthetic dataset.

## Folder Layout

Expected folders:

```text
dataset/
  train.jsonl
  validation.jsonl
  test.jsonl
  manual_merges.txt

vocab/
  vocab.json

binFiles/
  train_tokens.bin
  val_tokens.bin
  test_tokens.bin

model_checkpoints/
  checkpoint_latest.bin
  checkpoint_latest.optim
```

If `vocab/vocab.json` or the `.bin` files are missing, the program creates them from the dataset files.

For inference, place the downloaded weights here:

```text
model_checkpoints/checkpoint_latest.bin
model_checkpoints/checkpoint_latest.optim
vocab/vocab.json
```

## Compile

```bash
nvcc -std=c++17 -Iinclude -Iinclude/nlohmann \
  main.cpp attention.cpp embedding.cpp filehandler.cpp linear.cpp llmAssembly.cpp llmInference.cpp \
  moe.cpp optimizer.cpp residualADD.cpp rmsNorm.cpp rope.cpp SwiGLU.cpp tensor.cpp tokenUtility.cpp \
  cuda/adamW.cu cuda/crossEntropyLoss.cu cuda/embedding_lookup.cu cuda/gqa.cu cuda/linearmul.cu \
  cuda/matmul2d.cu cuda/moek.cu cuda/rootmeansquare_norm.cu cuda/ropek.cu cuda/silu.cu \
  cuda/tesnorKernal.cu \
  -o ironwillV1
```

## Train

```bash
./ironwillV1
```

Training resumes from the latest numbered checkpoint in `model_checkpoints/` when available.

## Inference

```bash
./ironwillV1 prompt $'[CALC]\nQuestion:\nDerivative of: sin x\n\nAnswer:'
```

Example output from IRONWILL_V1:

```text
[CALC] Question : Derivative of : sin x Answer : cos x
```

## Weights

Download the IRONWILL_V1 10K checkpoint package from the project release page:

```text
https://github.com/KshitijxFrl/irwWEB/releases/tag/ironwill-v1-10k
```

Put the files in:

```text
model_checkpoints/
vocab/
```

## Threminent

IRONWILL is built on the first iteration of **Threminent**, a lightweight C++/CUDA GPU framework:

```text
https://github.com/KshitijxFrl/Threminent
```
