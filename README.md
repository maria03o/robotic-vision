# M3 Vision — KukaEnv Observation Pipeline

Converts a PyBullet KukaEnv render into a fixed-size observation tensor for the RL team's PPO policy. Current version: **Week 2** (ResNet18 + physics state). Week 1 (FPN-based, no physics) is preserved below as project history.

---

## Table of Contents
- [Overview](#overview)
- [Current Architecture (Week 2)](#current-architecture-week-2)
- [Physics Dropout — Design Rationale](#physics-dropout--design-rationale)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Run](#run)
- [Usage in Your Own Code](#usage-in-your-own-code)
- [Observation Tensor Specification](#observation-tensor-specification)
- [Project History: Week 1 → Week 2](#project-history-week-1--week-2)
- [Phase 3 Ablation Plan (W5–W6)](#phase-3-ablation-plan-w5w6)
- [Handoff Notes for the RL Team](#handoff-notes-for-the-rl-team)

---

## Overview

The pipeline renders an RGB frame (plus depth and segmentation) from a KukaEnv PyBullet scene, extracts visual and physics features, and concatenates them into a single tensor that plugs directly into the PPO policy. The design goal throughout has been to prevent **shortcut learning**: without care, a policy with access to exact physics state (joint angles, end-effector position) has no incentive to actually learn from the visual scene. Two mechanisms address this — colour/crop augmentation on the visual path, and randomized physics-state dropout per rollout.

---

## Current Architecture (Week 2)

Locked per Selma's Week 2 feedback:

| Item | Decision |
|---|---|
| Visual backbone | **ResNet18 (512-dim)** — replaces Week 1's FPN backbone |
| Physics state | **(9,)** — 7 joint angles + end-effector (x, y) |
| Shortcut-learning mitigation | Random colour jitter + crop augmentation, every frame |
| Physics dropout rate | **30%** chance per rollout to zero out physics features |
| Final observation | **546-dim** = 512 (visual) + 25 (detection) + 9 (physics) |

```
RGB (224×224×3) ────────────────────────────────────────────────────────────┐
                                                                             │
 ┌─ Step 1 ──────┐  ┌─ Step 2 ─────────┐  ┌─ Step 3 ───────────────────┐    │
 │ KukaEnv render│  │ Physics state    │  │ ResNet18 + augmentation    │    │
 │ RGB / D / Seg  │─▶│ (9,) 7J + EE(x,y)│  │ (512,) — anti-shortcut     │    │
 └───────┬────────┘  └──────────────────┘  └─────────────────────────────┘    │
         │                                                                    │
 ┌─ Step 4 ─────────────────────┐                                            │
 │ Seg mask → bounding boxes    │                                            │
 │ det_vec (25,) — 5 objects    │                                            │
 └───────────────────────────────┘                                           │
                                                                             │
 ┌─ Step 5 ─────────────────────────────────────────────────┐                │
 │ Physics dropout mask ~ Bernoulli(0.7), sampled ONCE       │               │
 │ per rollout and held fixed for its full duration          │               │
 └────────────────────────────────────────────────────────────┘              │
                                                                             │
 ┌─ Step 6 ────────────────────────────────────────────────────────────┐     │
 │ obs = cat([ vis_feat(512) | det_vec(25) | physics * mask(9) ])      │     │
 │      → (546,)                                                       │     │
 └───────────────────────────────────────────────────────────────────────┘   │
                                                                             │
 ────────────────────────────────────────────────────────────────────────────▶ RL policy
```

---

## Physics Dropout — Design Rationale

If the policy can always rely on exact joint angles and end-effector position, it has no incentive to understand the visual scene. To force genuine visual grounding:

- At each **rollout start**, sample `mask ~ Bernoulli(0.7)`:
  - `mask = 1.0` (70% of rollouts) → physics state included.
  - `mask = 0.0` (30% of rollouts) → physics state zeroed out (vision-only).
- The mask is **fixed for the entire rollout**, so the policy can adapt its strategy for that rollout rather than being surprised at each step.
- At **evaluation time**, `mask = 1.0` always, giving a clean, consistent metric across runs.

---

## Project Structure

```
m3_vision/
├── src/
│   ├── env_render.py       # Step 1 — PyBullet KukaEnv scene rendering
│   ├── physics_state.py    # Step 2 — (9,) physics state vector
│   ├── visual_features.py  # Step 3 — ResNet18 + anti-shortcut augmentation
│   ├── detection.py        # Step 4 — seg mask → (25,) detection vector
│   └── observation.py      # Steps 5 & 6 — dropout mask + (546,) obs tensor
├── config.py                # WIDTH, HEIGHT, MAX_OBJECTS, DEVICE, dims, dropout rate
├── pipeline.py               # High-level convenience wrapper — run_pipeline()
├── main.py                   # Entry point + policy handoff check
├── smoke_test.py              # Shape assertions / sanity check
├── requirements.txt
└── README.md
```

---

## Setup

Python 3.10+ recommended. CUDA optional — falls back to CPU automatically.

```bash
git clone https://github.com/<your-username>/m3_vision.git
cd m3_vision
pip install -r requirements.txt
```

---

## Run

```bash
python smoke_test.py
```

```bash
python main.py
```

Expected output:

```
==================================================
M3 Vision — Week 2 Pipeline (KukaEnv Integration)
==================================================
[render]     rgb=(224, 224, 3)  depth=(224, 224)  seg=(224, 224)
[physics]    physics_state=(9,)
[backbone]   ResNet18 loaded
[detection]  det_vec=(25,)
[dropout]    rollout mask=1.0  (physics included)
[obs]        vis_feat=(512,)  det_vec=(25,)  physics=(9,)
[obs]        observation=(546,)

WEEK 2 CHECK PASSED
  torch.Size([546])  →  dummy policy  →  torch.Size([1, 7])
```

---

## Usage in Your Own Code

```python
from src.visual_features import load_backbone
from src.observation import build_observation

backbone = load_backbone()   # call once at startup

# per frame:
observation, vis_feat, det_vec, physics_masked = build_observation(
    rgb, depth, seg, physics_state, backbone
)
# observation.shape → (546,)  ready for the PPO policy
```

Or run the full pipeline, including rendering and rollout-level dropout sampling:

```python
from pipeline import run_pipeline

observation = run_pipeline(verbose=False)
```

---

## Observation Tensor Specification

| Component | Shape | Source | Notes |
|---|---|---|---|
| Visual features (ResNet18) | `(512,)` | RGB → augmented → backbone | — |
| Detection vector | `(25,)` | Seg mask → bounding boxes, 5 object slots | 5 features/object: `[x1, y1, x2, y2, mean_depth]` |
| Physics state (masked) | `(9,)` | 7 joint angles + EE (x, y) | Zeroed in 30% of rollouts |
| **Observation** | **(546,)** | Concatenated | **RL policy input** |

---

## Project History: Week 1 → Week 2

### Week 1 — Initial Pipeline (superseded)

The first version used a Faster R-CNN **FPN backbone** (256-dim, global-average-pooled from the `"0"` feature map) with no physics state, giving a 281-dim observation:

```
RGB (224×224×3) ──► FPN backbone ──► visual features (256,) ─┐
depth + seg mask ──► bounding boxes ──► detection vector (25,) ─┴──► observation (281,)
```

| Component | Week 1 shape |
|---|---|
| Visual features (FPN) | `(256,)` |
| Detection vector | `(25,)` |
| **Observation** | **(281,)** |

The Week 1 layout used flat module files (`config.py`, `render.py`, `boxes.py`, `backbone.py`, `observation.py`, `pipeline.py`, `main.py`) rather than the current `src/` package layout.

### What Changed in Week 2

| Change | Week 1 | Week 2 |
|---|---|---|
| Visual backbone | Faster R-CNN FPN | **ResNet18** |
| Visual feature dim | 256 | **512** |
| Physics state | none | **(9,)** — 7 joints + EE (x, y) |
| Shortcut mitigation | none | **Colour jitter + crop augmentation**, plus **30% physics dropout** |
| Observation dim | 281 | **546** |
| Module layout | Flat files | `src/` package, one file per pipeline step |

If you're working from an older checkout, `boxes.py` → `detection.py`, `backbone.py` → `visual_features.py`, and `render.py` → `env_render.py`; the function signatures were also updated to accept and return the physics state and dropout mask.

---

## Phase 3 Ablation Plan (W5–W6)

| Ablation | Change | Hypothesis |
|---|---|---|
| **A: FPN** | Swap ResNet18 → Faster R-CNN FPN backbone | Do detection-tuned features help? |
| **B: SigLIP** | Swap to SigLIP vision encoder | Does CLIP-style training aid language grounding? |
| **C: DINOv2** | Swap to DINOv2 ViT-S/14 | Do self-supervised features generalize better? |
| **D: No physics** | Permanently zero the physics state | How much does proprioception actually matter? |

All ablations share the same `build_observation()` interface — only the backbone (or physics-masking behavior, for ablation D) changes, so no changes are needed downstream in the RL policy code.

---

## Handoff Notes for the RL Team

- The observation tensor contract is fixed at **(546,) float32**, assembled on `DEVICE` from `config.py`.
- `mask` behavior differs between training and eval — see [Physics Dropout](#physics-dropout--design-rationale). Make sure eval runs use `mask = 1.0` for comparable metrics across ablations.
- Swap the placeholder policy call in `main.py` for the real PPO policy:

```python
# main.py
action = real_ppo_policy(observation.unsqueeze(0))
```

- If integrating an ablation from Phase 3, only the backbone module needs to change — `build_observation()`'s interface and the `(546,)` output shape stay identical (except for ablation D, which zeroes the physics slice permanently rather than stochastically).
