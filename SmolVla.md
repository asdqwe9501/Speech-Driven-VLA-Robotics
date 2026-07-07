# SmolVLA: Transferring Vision-Language Models to Robot Control

> Reference notes for the SO-ARM 101 project — from core concepts to implementation and real robot integration

**Paper:** SmolVLA (arXiv:2406.xxxxx) · **Official implementation:** [huggingface/lerobot](https://github.com/huggingface/lerobot)
**Last updated:** 2026-06-21

---

## Table of Contents

1. [SmolVLA at a Glance](#1-smolvla-at-a-glance)
2. [Core Concepts](#2-core-concepts)
3. [Model Architecture](#3-model-architecture)
4. [Training Methodology](#4-training-methodology)
5. [Implementation Guide](#5-implementation-guide)
6. [SO-ARM 101 Integration](#6-so-arm-101-integration)
7. [Challenges & Solutions](#7-challenges--solutions)
8. [References](#8-references)

---

## 1. SmolVLA at a Glance

SmolVLA is a technique for **transferring the knowledge of a large, capable vision-language model (VLM) into a small, fast robot policy network**.

Large VLMs like CLIP are trained on hundreds of millions of image-text pairs from the internet, so they already have a strong understanding of "what something is, where it is, and how it can be manipulated." The problem is that these models are too large to run directly on a robot. SmolVLA uses knowledge distillation to compress this understanding into a lightweight form and combines it with imitation-learning frameworks such as ACT or Diffusion Policy, producing a complete image-to-robot-action pipeline.

### Key Contributions

| Contribution | Description |
|---|---|
| **Distillation-based compression** | Transfers features from large VLMs (CLIP, LLaVA) into a small policy model |
| **Proven manipulation performance** | Competitive results vs. large baseline models on pick-and-place, insertion, and in-hand manipulation |
| **Efficient transfer pipeline** | VLM encoder frozen, only the policy head is trained → fast learning from limited data |
| **Edge deployment potential** | Reduced compute enables real-time control on onboard robot computers (Jetson-class hardware) |
| **Compatibility with policy frameworks** | Works with ACT, Diffusion Policy, and other imitation-learning approaches |

### Why It Matters

Traditional robot learning trains the vision encoder from scratch using only limited robot data, which is data-inefficient. SmolVLA instead reuses the visual understanding already learned by internet-scale VLMs, so **a strong policy can be learned from relatively few demonstrations**. This is especially valuable for low-cost arms like SO-ARM — near-real-time, vision-based manipulation becomes feasible even without an onboard GPU and with battery-powered constraints.

---

## 2. Core Concepts

SmolVLA is four components working together: **Eyes (Vision) → Bridge (Adapter) → Brain (Language Model) → Hands (Action Head)**.

| Component | Role | Trainable? |
|---|---|---|
| CLIP Vision Encoder | Turns camera images into meaningful features | Frozen |
| Adapter | Translates between vision and language feature spaces | **Trainable** |
| LLaMA Language Model | Reasons over features to decide the next action | Frozen |
| Action Head | Converts predicted tokens into continuous robot commands | **Trainable** |

**The trick:** the Vision Encoder and Language Model are large and capable but kept frozen; only the small adapter and action head are trained. This means only about **0.2% of all parameters** are updated during fine-tuning, yet performance remains strong.

### 2.1 Knowledge Distillation — "Learning from a Teacher"

- **Large model**: very capable, but slow (introduces lag in robot reactions)
- **Small model**: fast, but not naturally as capable

Distillation trains the small model to mimic the large model's judgments (e.g., "this is a red ball I can grasp, located in the upper-left area"), producing a model that is both small and capable.

$$\text{Total loss} = 0.8 \times \text{match real actions} + 0.2 \times \text{mimic the large model}$$

### 2.2 Vision Encoder — "The Eyes"

Converts the camera image into a meaningful feature vector.

- Split a 224×224 image into 196 patches of 16×16
- Convert each patch into a 768-dimensional vector
- Aggregate everything into a whole-image representation
- Result: "a red ball, round, medium-sized, graspable"

Because CLIP was pretrained on hundreds of millions of internet image-text pairs, it can reasonably understand objects the robot has never explicitly been trained on. Keeping this component **frozen** is essential — it preserves CLIP's already-good representations instead of letting them degrade during robot fine-tuning.

### 2.3 Language Model — "The Brain"

Takes the information extracted by the Vision Encoder and sequentially reasons about "what to do next." It's originally a language-generation model (LLaMA), repurposed in SmolVLA to **generate robot action tokens** instead. Because it's a sequence model, previous actions naturally influence the next one, and its pretrained language understanding lets it also process text instructions (e.g., "pick up the red block").

### 2.4 Adapter — "The Bridge"

Vision (768D) and the Language Model (512D or 4096D) speak different "languages." The adapter translates between them.

```
Input (768D) → Linear Down (768→64, bottleneck) → ReLU → Linear Up (64→768) → + skip connection → Output
```

Thanks to the small bottleneck, the adapter has relatively few parameters (tens of thousands), which lets it train quickly on limited robot data without overfitting.

### 2.5 Training Loss Function

```
Total loss = action loss + 0.1 × distillation loss (mimic large model)
```

- **Action loss**: matching the actual robot action (primary objective)
- **Distillation loss**: mimicking the teacher model's prediction (auxiliary)

### 2.6 Evaluation Criteria

| Metric | Target |
|---|---|
| Offline action accuracy | > 85% |
| Online task success rate | > 70% |
| Inference speed | < 50ms (relative to control cycle) |

---

## 3. Model Architecture

### 3.1 End-to-End Pipeline: Training vs. Inference

![SmolVLA training vs inference pipeline](./images/smolvla-training-inference.svg)

The forward computation graph is **identical** during training and inference — that's a key difference from CVAE-based policies like ACT. Both sides run: raw RGB image + robot joint state → frozen CLIP encoder → trainable adapter → frozen LLaMA decoder → trainable action head → predicted action chunk. An optional text instruction can be injected alongside the visual embedding.

The only difference is what happens with the output:
- **Training**: the predicted action chunk is compared against the expert demonstration (ground-truth actions from teleoperation) to compute a loss. Gradients flow back only into the adapter and action head — CLIP and LLaMA stay frozen throughout.
- **Inference**: there is no ground truth available. The predicted action chunk is sent straight to the robot for execution.

### 3.2 Vision Encoder in Detail

![CLIP vision encoder step by step](./images/vision-encoder.svg)

| Property | ViT-Base | ViT-Large |
|---|---|---|
| Input image | 224×224 | 224×224 or 336×336 |
| Patch size | 16×16 | 16×16 |
| Number of patches | 196 | 196 or 441 |
| Embedding dimension | 768 | 1024 |
| Transformer layers | 12 | 24 |
| Attention heads | 12 | 16 |
| Parameters | ~86M | ~303M |

**Pretraining data:** ImageNet-21K + LAION image-text pairs (400M+), using contrastive learning to maximize image-text alignment.

**Freezing strategy comparison**

| Strategy | Memory | Training time | Overfitting risk | Generalization | Recommendation |
|---|---|---|---|---|---|
| Fully frozen | Minimal | Fast | Low | High | ✅ SmolVLA default |
| Partial fine-tuning | Medium | Medium | Medium | Medium | Only with ample data |
| Full fine-tuning | High | Slow | High | Low | Not recommended |

With full freezing, memory usage drops from ~10GB to ~2GB (~80% reduction), and training time drops from 8 hours to 1 hour (~8× faster).

### 3.3 Adapter Module

![Adapter bottleneck design](./images/adapter-detail.svg)

**Parameter efficiency**

| Component | Parameters | Trainable | Share |
|---|---|---|---|
| CLIP ViT-Base | 86M | ✗ Frozen | 0% |
| Adapter (down/up) | ~100K | ✓ | ~0.1% |
| LLaMA-7B | 7B | ✗ Frozen | 0% |
| Action head | 50–100K | ✓ | ~0.15% |
| **Total trainable** | **~150K** | – | **~0.2%** |

### 3.4 Language Model (LLaMA) Details

| Property | Setting |
|---|---|
| Model size | 1.3B / 7B / 13B / 70B |
| Hidden dimension | 4096 (7B) / 5120 (13B) |
| Transformer layers | 32 (7B) / 40 (13B) |
| Vocabulary | 32,000 |
| Context window | 2048 tokens |
| Positional embedding | RoPE |
| Normalization | RMSNorm |

The robot's continuous control signal is converted into discrete tokens via **Vector Quantization (VQ)**, fed into the language model, and generated autoregressively one token at a time.

### 3.5 Action Head & Action Chunking

The action head converts LLaMA's hidden states back into real control signals: `Linear → Softmax → token selection → inverse VQ → denormalization`.

SmolVLA can predict **4–8 steps of actions at once** from a single forward pass (action chunking). This reduces inference latency and makes robot motion more stable, but predicting further into the future can accumulate error, so 1–4 step chunking is generally recommended.

### 3.6 Why CLIP + LLaMA?

| Aspect | Direct network (CNN→action) | Pure RL | Full-size teacher model | **SmolVLA** |
|---|---|---|---|---|
| Data efficiency | Poor | Worst | Good | **Best** |
| Memory | Small | Small | Huge (200GB+) | **Small (~4GB)** |
| Latency | Fast | Fast | Slow (500ms+) | **Medium (optimizable)** |
| Performance | Medium | Low | Best | **High** |
| Practicality score | 40/100 | 30/100 | 60/100 | **85/100** |

**Three design principles:** ① maximize use of pretraining (CLIP's visual understanding + LLaMA's reasoning), ② partial fine-tuning (freeze encoder and LM, train only adapter and action head), ③ computational efficiency (trainable on a 4GB GPU, deployable at the edge).

---

## 4. Training Methodology

### 4.1 Overall Training Workflow

![Training workflow: data collection to deployment](./images/training-workflow.svg)

### 4.2 Data Preparation

| Item | Value |
|---|---|
| Source | Teleoperation or manual demonstration |
| Camera | RGB, mounted top-down / side / on wrist |
| Sampling rate | 20fps (50ms intervals) |
| Demo length | 10–30 seconds (200–600 frames) |

**Preprocessing pipeline:** downsample to 20fps → ImageNet normalization (mean/std) → resize to 224×224 → normalize actions ([-π, π] → [-1, 1]) → (optional) color jitter / rotation / translation augmentation.

**Performance by dataset size**

| Scenario | Samples | Training time | Performance |
|---|---|---|---|
| Small (1–2 tasks) | 1K–5K | 10–20 min | 70–80% |
| Medium (3–5 tasks) | 10K–30K | 1–2 hours | 80–85% |
| Large (10+ tasks) | 50K–100K+ | 4–8 hours | 85–95% |

### 4.3 Training Configuration

```python
optimizer = AdamW(trainable_params, lr=1e-4~5e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)
warmup_steps = 500~1000

loss = 0.9 * CrossEntropyLoss(action_logits, action_targets) \
     + 0.1 * KLDivergence(student_output, teacher_output, temperature=3.0)
```

| Parameter | Recommended | Range |
|---|---|---|
| Batch size | 64 | 32–128 |
| Learning rate | 1e-4 | 5e-5–5e-4 |
| Epochs | 10–30 | – |
| Early stopping | Stop if no improvement for 2000 steps | – |
| Gradient clip | 1.0 | – |
| Mixed precision | FP16 | – |

### 4.4 Convergence Behavior

Loss drops sharply during the first 1K steps (most of the gain), improves gradually from 1K–5K steps (fine detail), and converges/stabilizes from 5K–20K steps.

| Approach | Training time | Data required | Final performance |
|---|---|---|---|
| Fine-tuning (SmolVLA) | 1–8 hours | 5K–20K | 85–90% |
| Training from scratch | 24–72 hours | 50K–100K+ | 90–95% |

### 4.5 Inference Latency

| Stage | Time (on A100) |
|---|---|
| CLIP Vision Encoder | 30–50ms |
| Adapter | 1–2ms |
| LLaMA forward (1 token) | 100–200ms |
| **Total (single action)** | **150–250ms** |

This is far slower than a typical robot control cycle (50Hz = 20ms), so real-time control requires optimization:

- **Action chunking**: predicting 4 steps at once brings effective latency down to 100ms ÷ 4 = 25ms
- **INT8 quantization**: roughly 3× faster than FP32 (1–3% accuracy loss)
- **KV caching**: roughly 2× faster autoregressive generation
- **Edge deployment**: Jetson AGX Orin (275 TFLOPS, ~1/10 the cost and ~1/20 the power of an A100)

Combining action chunking (8 steps) with quantization brings effective latency down to roughly 6–12ms, comfortably within the 20ms control cycle.

---

## 5. Implementation Guide

### 5.1 Official Implementation: LeRobot (Hugging Face)

```bash
pip install lerobot
# or install from source
git clone https://github.com/huggingface/lerobot
cd lerobot && pip install -e .
```

**Requirements**: `torch>=2.0`, `transformers>=4.36`, `diffusers>=0.24`, `huggingface-hub>=0.19`, `lerobot>=0.1`, `opencv-python-headless`, `pillow`, `tensorboard`

### 5.2 Loading the Model

```python
import torch
from lerobot.common.policies.smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained(
    "facebook/smolvla-base",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
policy.eval()
```

### 5.3 Input/Output Format

- **Image input**: `(B, 3, H, W)`, 224×224 or 256×256, float32, ImageNet-normalized
- **Text prompt**: string, up to 77 tokens (CLIP standard)
- **Action output**: `(B, T, 7)` — `[0:6]` joint angles (−π to π), `[6]` gripper (0 to 1)

```python
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor = transform(Image.open("robot_view.jpg").resize((224, 224))).unsqueeze(0).to("cuda")

with torch.no_grad():
    action = policy(image=image_tensor, text="Pick up the red cube")

print(action[0, :6])  # joint angles
print(action[0, 6])   # gripper
```

### 5.4 Running Training

```bash
python -m lerobot.scripts.train \
    --config-name smolvla-base \
    --dataset-repo-id huggingface/lerobot-xarm-lift \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --num-steps 50000 \
    --output-dir ./checkpoints/smolvla-finetuned
```

| Parameter | Recommended | Range |
|---|---|---|
| Learning rate | 1e-4 | 5e-5–5e-4 |
| Batch size | 64 | 32–128 |
| Gradient accumulation | 4 | 2–8 |
| Warmup steps | 1000 | 500–2000 |
| Max steps | 50000 | 30000–100000 |

### 5.5 Evaluation

```python
def evaluate_task_success(model, test_episodes, success_fn, num_rollouts=10):
    success_count, total_count = 0, 0
    for episode in test_episodes:
        for _ in range(num_rollouts):
            obs, prompt, success = episode["initial_observation"], episode["task_prompt"], False
            for step in range(episode["max_steps"]):
                with torch.no_grad():
                    action = model(image=obs["image"], text=prompt)
                obs = episode["environment"].step(action.numpy())
                if success_fn(obs):
                    success = True
                    break
            total_count += 1
            success_count += int(success)
    return success_count / total_count * 100
```

### 5.6 Performance Optimization

| Technique | Method | Effect |
|---|---|---|
| Gradient checkpointing | `policy.enable_gradient_checkpointing()` | ~50% memory reduction |
| Mixed precision | `torch.cuda.amp.autocast()` + `GradScaler` | Lower memory, faster training |
| Action chunking | `action_chunk_size=4~8` | Lower latency |
| INT8 quantization | `torch.quantization.quantize_dynamic` | 2–4× faster inference |
| ONNX export | `torch.onnx.export(...)` | Production deployment |

### 5.7 Common Issues

| Issue | Cause | Fix |
|---|---|---|
| CUDA out of memory | Batch size too large | Reduce batch size + increase gradient accumulation |
| NaN loss | Unstable learning rate | Lower LR (1e-4→5e-5), add gradient clipping |
| Not converging | Learning rate too high | Reduce LR by 10× |
| Overfitting | Insufficient data | Data augmentation, early stopping, more dropout |
| Slow inference | No batching/chunking | Enable action chunking, apply quantization |
| Action out of range | Normalization bug | Re-check normalization logic, add clipping |

### 5.8 Implementation Checklist

- [ ] `pip install lerobot`, torch 2.0+, transformers 4.36+
- [ ] Load pretrained model (`facebook/smolvla-base`)
- [ ] Input normalization (ImageNet mean/std)
- [ ] Prepare dataset (images, actions, text prompts)
- [ ] Set hyperparameters (LR 1e-4, batch 64, 50k steps)
- [ ] Evaluate (MAE, RMSE, task success rate)
- [ ] Optimize (quantization, action chunking, ONNX export)

---

## 6. SO-ARM 101 Integration

### 6.1 Hardware Specifications

| Item | Spec |
|---|---|
| Degrees of freedom | 6-DOF (6 rotational joints) + gripper |
| Payload | Up to 1–3kg |
| Repeat precision | ±0.5mm |
| Workspace | ~900mm |
| Control cycle | 50Hz (20ms) or 100Hz (10ms) |
| Gripper | 2-finger parallel, 0–100mm stroke, ~200–300ms response |

**Joint layout**: Base rotation (±π) → Shoulder (±π/2) → Elbow (±π) → Wrist 1 (±π) → Wrist 2 (±π/2) → Wrist 3 (±π)

**Recommended camera setup**: 640×480 or 1280×720, 30–60fps, 50–70° field of view, USB 3.0 or networked camera (e.g. Intel RealSense D435, Logitech webcam).

### 6.2 Action Space Mapping

SmolVLA outputs a normalized 7-dimensional vector in [-1, 1], which must be converted into SO-ARM's actual radian/gripper values.

```python
import numpy as np
from typing import Dict

class ActionMapper:
    """Converts SmolVLA actions [-1,1] to/from SO-ARM control signals (radians, 0-1)"""

    def __init__(self):
        self.joint_ranges = np.array([
            [-np.pi, np.pi],      # Joint 1: Base
            [-np.pi/2, np.pi/2],  # Joint 2: Shoulder
            [-np.pi, np.pi],      # Joint 3: Elbow
            [-np.pi, np.pi],      # Joint 4: Wrist 1
            [-np.pi/2, np.pi/2],  # Joint 5: Wrist 2
            [-np.pi, np.pi],      # Joint 6: Wrist 3
        ])

    def map_vla_to_soarm(self, vla_action: np.ndarray) -> Dict:
        """[-1,1] normalized action -> actual joint angles (rad) + gripper (0-1)"""
        joint_angles = np.zeros(6)
        for i in range(6):
            min_a, max_a = self.joint_ranges[i]
            normalized = np.clip((vla_action[i] + 1.0) / 2.0, 0, 1)
            joint_angles[i] = min_a + normalized * (max_a - min_a)

        gripper = np.clip((vla_action[6] + 1.0) / 2.0, 0, 1)
        return {'joint_angles': joint_angles, 'gripper': float(gripper)}

    def clip_to_limits(self, action_dict: Dict) -> Dict:
        """Safety clipping to joint/gripper limits"""
        joint_angles = np.clip(action_dict['joint_angles'],
                                self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        gripper = np.clip(action_dict['gripper'], 0, 1)
        return {'joint_angles': joint_angles, 'gripper': float(gripper)}
```

When preparing training data, the reverse conversion is needed (actual joint angles → SmolVLA normalized format):

```python
def denormalize_to_vla(robot_state: Dict, joint_ranges: np.ndarray) -> np.ndarray:
    """Actual robot state -> SmolVLA normalized action format (for training data prep)"""
    action_vla = np.zeros(7)
    for i in range(6):
        min_a, max_a = joint_ranges[i]
        normalized = (robot_state['joint_angles'][i] - min_a) / (max_a - min_a)
        action_vla[i] = normalized * 2.0 - 1.0
    action_vla[6] = robot_state['gripper'] * 2.0 - 1.0
    return np.clip(action_vla, -1, 1)
```

### 6.3 Image Preprocessing

Since CLIP was pretrained on ImageNet, the same normalization is required: BGR→RGB → resize to 224×224 → scale to 0–1 → ImageNet mean/std normalization. (Wrap this in an `SO_ARM_ImageProcessor` class to reuse for every live frame.)

### 6.4 Demonstration Collection

Teleoperation records image, action, and timestamp data together. Recommended directory layout:

```
demonstrations/
├── task_001_pick_and_place_red_cube/
│   ├── video.mp4
│   ├── actions.npy        # (T, 7)
│   └── metadata.json
├── task_002_grasp_small_object/
...
```

**Collection guidelines**

| Item | Recommendation |
|---|---|
| Demos per task | At least 50, ideally 100–200 |
| Total frames | 10K–100K |
| Success rate | Only record successful demonstrations |
| Diversity | Vary starting position, speed, and lighting |
| Frame rate | Collect at 30fps → downsample to 20fps |

### 6.5 Fine-Tuning Strategy Comparison

| Strategy | Training time | Memory | Performance | Recommended for |
|---|---|---|---|---|
| Full fine-tuning | 8–24 hours | 20–30GB | 95%+ | Large datasets (100K+) |
| **LoRA fine-tuning** | 1–4 hours | 4–6GB | 90%+ | Medium datasets (10K–50K) — **recommended** |
| Frozen backbone + adapter only | 30 min–2 hours | 2–4GB | 85–90% | Small datasets (1K–10K) |

LoRA attaches low-rank adapters to the attention layers (`q_proj`, `v_proj`) and sets only the adapter and action head to `requires_grad=True`, balancing memory efficiency with performance.

### 6.6 Full System Integration

![SmolVLA and SO-ARM 101 real-time control loop](./images/so-arm-realtime-loop.svg)

The real-time control loop only invokes SmolVLA when a new action chunk is needed, executes multiple predicted steps sequentially, and keeps pace with the 20ms cycle. On a `KeyboardInterrupt`, calling `robot.stop()` to safely halt the arm is essential.

---

## 7. Challenges & Solutions

| Challenge | Problem | Solution |
|---|---|---|
| **Data collection burden** | High-quality demonstrations take time and cost | Start with easy tasks, automate success/failure detection, augment with color jitter/rotation/translation |
| **Sim-to-real gap** | Sensitive to lighting, camera, and object changes | Domain randomization (randomize brightness/color temperature/noise/blur), include ~20% real robot data, use Huber loss |
| **Inference latency** | 100–200ms inference vs. 20ms control cycle | Action chunking (4–8 steps), INT8 quantization, KV caching |
| **Limited generalization** | Works well only on trained tasks/objects | Train on diverse tasks simultaneously, leverage CLIP's pretrained representations, few-shot adaptation |

### Evaluation Plan

**Offline evaluation**: compute MAE/RMSE on a held-out test set (including per-joint breakdown)
**Online evaluation**: measure success rate over N trials on the real robot

| Task | Difficulty | Success criterion | Target success rate |
|---|---|---|---|
| Pick and Place | Easy | Object in gripper + reaches target position | 85%+ |
| Insertion | Medium | Object inserted into hole | 75%+ |
| In-hand Rotation | Medium | Object rotated 90° | 70%+ |
| Pushing | Easy | Object moved to target | 80%+ |
| Stacking | Hard | Two or more blocks stacked | 60%+ |

**Reference benchmark results** (from the paper, varies with training sample count): Pick and Place 82%, Insertion 75%, In-hand Rotation 70%, Pushing 80% — roughly 5–10 points below equivalent large-model baselines under the same conditions.

### Implementation Timeline (example, ~10 weeks)

1. **Preparation & environment setup** — understand the paper, set up LeRobot/PyTorch, basic SO-ARM control code, camera calibration
2. **Data collection & initial fine-tuning** — 10–20 demos, frozen-backbone fine-tuning, validate action mapping
3. **Scale up data & full fine-tuning** — 100–500 demos, LoRA fine-tuning, online testing
4. **Optimization** — INT8 quantization, action chunking, domain randomization
5. **Final evaluation & documentation** — performance evaluation, code cleanup, demo video

---

## 8. References

**Official implementation & paper**
- LeRobot GitHub: https://github.com/huggingface/lerobot
- LeRobot Hub: https://huggingface.co/lerobot
- SmolVLA model: https://huggingface.co/facebook/smolvla-base

**Model components**
- CLIP: https://huggingface.co/openai/clip-vit-base-patch32
- LLaMA 2: https://huggingface.co/meta-llama/Llama-2-7b

**Datasets**
- LeRobot Datasets: https://huggingface.co/datasets?search=lerobot
- XARM Lift: https://huggingface.co/datasets/huggingface/lerobot-xarm-lift

**Further reading**
- Hugging Face Transformers docs: https://huggingface.co/docs/transformers
- PyTorch official docs: https://pytorch.org/docs/stable/index.html

---

*This document was compiled from personal Obsidian notes (`summary.md`, `key-concepts.md`, `architecture.md`, `implementation-notes.md`, `so-arm-integration.md`).*
