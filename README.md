# Vision-Language-Action Robot Control with Mixture-of-Experts

## Current milestone

**Step 8 — DAgger / Corrective On-Policy Data**

Step 7 found that adding a short observation/action history to the Dense
VLA fixed gripper-timing oscillation almost completely (median switches
23.5 -> 1.0) but left closed-loop success essentially flat (22% vs
Dense's 24% at ±3cm), with the remaining failures (`failed_to_lift`,
`pushed_cube_away`) unchanged in proportion. The leading suspect was
**exposure bias**: Temporal is trained on expert trajectories (teacher
forcing) but deployed on states its own imperfect actions produce, which
the expert never demonstrated recovering from. Step 8 tests the direct
remedy: collect the states the Temporal policy actually visits during its
own rollouts, label them with a corrective expert action, add those
corrective pairs to the training data, and fine-tune.

**Headline measured result: the hypothesis is NOT supported by this
experiment -- DAgger fine-tuning made closed-loop success substantially
*worse*, not better, at every distance tested.** Closed-loop success:
Temporal+DAgger 4.0%/2.0%/2.0% at ±3/4/5cm vs. Temporal's frozen
22%/16%/14% (Dense 24%/18%/12%, MoE 0%/2%/4%) -- despite the corrective
data measurably improving the offline metric it directly targeted
(corrective-state joint MAE 0.056 -> 0.021, a 62% reduction). This is a
real, diagnosed regression, not a bug: see "Interpretation" for the
root-cause analysis (the corrective expert's LIFT-phase label does not
verify an actual grasp before committing, and that flawed label dominates
a large share of the retained corrective data).

```text
Temporal (Step 7, frozen baseline):        Temporal + DAgger (this milestone):

D_expert (Step 3 demonstrations)           D_expert  UNION  D_dagger
    |                                            |
    v                                            v
train Temporal Dense VLA               fine-tune Temporal Dense VLA
    |                                       ^         |
    v                                       |         v
closed-loop rollout                    corrective    closed-loop rollout
    |                                    expert            |
    v                                       ^               v
visited states  -------------------------->|          re-evaluate
(labeled offline, never executed)
```

### Recap: Steps 4-7, frozen baselines

Step 4 trained the Dense VLA and validated it offline (near-perfect
action-prediction accuracy on held-out data). Step 5 asked the question
offline metrics cannot answer: does the policy actually work when it has
to act on its own observations, in real time, with no expert in the loop?
The Dense checkpoint (`outputs/training/dense_vla_run_001/best.pt`) and
its Step 5 closed-loop numbers below are **frozen experimental evidence**
for this milestone -- not retrained, not modified.

```text
MuJoCo
   |
   v
SimulationEnvironment
   |
   v
Observation_t (RGB_t + state_t)  +  language instruction
   |
   v
DenseVLAPolicy.predict()   <-- ONLY these two inputs; see below
   |
   v
RobotAction_t
   |
   v
env.step(action_t)
   |
   v
Observation_t+1  ---(loop)---
```

**Unlike the scripted expert, the Dense VLA receives no privileged cube
position, Jacobian, or controller stage** -- `ScriptedController` is never
imported or consulted during a learned rollout
(`evaluation/closed_loop.py`, enforced by
`tests/test_no_privileged_vla_inputs.py`). Task success is still measured
by the same physical sustained-cube-lift detector used since Step 2.5/3,
never by a model "done" signal (the model doesn't have one).

**Measured result: raw closed-loop success is far below offline accuracy**
(24% at the training distribution vs. ~99.95% offline gripper accuracy /
0.0029 rad offline joint MAE) -- see "Closed-loop results" below for the
full breakdown and failure analysis. This is a real, diagnosed finding,
not a bug: near-perfect one-step imitation does not imply a well-behaved
closed loop, because small per-step errors compound as the policy acts on
states its own earlier (slightly wrong) actions produced.

Step 6 (Sparse MoE) and Step 7 (Temporal Dense, observation/action
history) are likewise **frozen experimental evidence** for this milestone.
Step 8 fine-tunes *from* the Step 7 checkpoint
(`outputs/training/temporal_dense_vla_run_001/best.pt`) but never
modifies it -- see "Preserve all existing baselines" below.

### Dense VLA architecture (from Step 4, unchanged)

```text
RGB ------------> VisionEncoder (frozen ResNet18)  ---\
                                                        \
Language -------> LanguageEncoder (frozen DistilBERT) --+--> MultimodalFusion --> [4 tokens] --> Dense Transformer --> action_query --> ActionHead --> 8D action
                                                        /
23D State -------> StateEncoder (MLP, trained) --------/
```

| Component | Choice | Trainable? |
|---|---|---|
| Vision encoder | `torchvision` ResNet18, ImageNet-pretrained, pooled 512D output | No (frozen, eval mode) |
| Language encoder | HuggingFace `distilbert-base-uncased`, mean-pooled 768D output | No (frozen, eval mode) |
| State encoder | MLP `23 -> 128 -> 256`, GELU + LayerNorm | Yes |
| Fusion | Each modality projected to `hidden_dim`; token sequence `[VISION, LANGUAGE, STATE, ACTION_QUERY]` + learned per-slot embedding | Yes |
| Backbone | `nn.TransformerEncoder`, dense FFN only (no MoE) | Yes |
| Action head | MLP `hidden_dim -> hidden_dim -> 8` (7 joint targets + 1 gripper logit) | Yes |

Default `DenseVLAConfig`: `hidden_dim=256, num_layers=4, num_heads=8,
ffn_dim=1024, dropout=0.1, max_instruction_length=32`. All Transformer FFNs
are standard dense layers -- this is intentionally the future MoE
milestone's dense baseline (`--train-vision-encoder` /
`--train-language-encoder` unfreeze the pretrained backbones if ever
needed, both default `False`).

### Normalization

Both the 23D state and the 7 joint targets are mean/std normalized using
statistics computed **from the train split only**
(`training/normalization.py::fit_normalizers_from_split`) -- val/test
statistics never leak into training, and near-zero-std dimensions fall
back to `std=1` instead of dividing by ~0. Both normalizers are saved
inside every checkpoint so inference always uses the exact stats training
used. The gripper target is *not* normalized -- it's already `[0, 1]` and
treated as a binary open/closed label.

### Loss

```text
L = MSE(pred_joints_normalized, gt_joints_normalized)
  + GRIPPER_LOSS_WEIGHT * BCEWithLogitsLoss(pred_gripper_logit, gt_gripper)
```

`GRIPPER_LOSS_WEIGHT` defaults to `1.0` (`--gripper-loss-weight`). BCE (not
MSE) on a raw logit was chosen because expert gripper behavior is
effectively binary open/closed.

### Model input/output contract

- **Model input**: RGB frame, 23D `RobotState.as_vector()`, instruction
  string. Nothing else -- `DenseVLA.forward()`'s signature only accepts
  `pixel_values`, `input_ids`, `attention_mask`, `state`
  (`tests/test_model_shapes.py::test_model_input_contract_excludes_privileged_information`).
  Cube ground truth, the Jacobian, and controller stage never reach the
  model, exactly as they never reach `Observation.state` (Step 2.5/3).
- **Model output**: 7 joint targets + 1 gripper target, denormalized back
  to the same `RobotAction` used everywhere else in the project
  (`models/policy.py::DenseVLAPolicy.predict()` returns a real
  `RobotAction`, constructed nowhere else).

### Dataset (from Step 3, unchanged this milestone)

```text
data/demonstrations/
├── successful/episode_000000/{rgb/*.png, trajectory.npz, metadata.json}
├── manifest.json   (episode/sample counts, dims, seed)
└── splits.json     (episode-level train/val/test split, no overlap)
```

100 successful episodes, ~21k timesteps, `state_dim=23`, `action_dim=8`,
instruction is one of 4 semantically-identical variants stored once per
episode. Generated entirely from the MuJoCo scripted expert (Step 2.5),
never from a physical robot. Regenerate with
`python -m dataset.generate_dataset --episodes 100 --seed 42
--xy-randomization 0.03 --overwrite`; validate with `python -m
dataset.validation data/demonstrations`. Full schema/generation details
are in `dataset/` module docstrings.

### Offline results (Step 4, `outputs/training/dense_vla_run_001`, 30 epochs, seed 42)

```text
Parameters: total=81,198,408  trainable=3,659,016
Tiny overfit (64 samples, 150 epochs): joint loss 0.555 -> 0.011 (98% reduction), gripper acc 100%
Best validation (epoch 27): joint MAE 0.0029 rad, gripper accuracy 99.9%
Held-out TEST split (2,159 samples): joint MAE 0.0029 rad, gripper accuracy 99.95%
Per-joint MAE (test, rad): [0.0006, 0.0061, 0.0011, 0.0043, 0.0003, 0.0067, 0.0015]
Batch-1 inference latency: mean 7.7ms / p50 7.9ms on MPS (Apple M5 Max)
```

### Closed-loop results (Step 5, same checkpoint, `outputs/evaluation/dense_vla_closed_loop/`)

Raw policy, no smoothing, control cadence unchanged from data generation
(`control_substeps=10`), 50 episodes per condition unless noted:

```text
Distribution shift (cube XY randomization), seed 42:
  +/-3cm (training distribution): 12/50  24.0%
  +/-4cm (mild OOD):                9/50  18.0%
  +/-5cm (mild OOD):                6/50  12.0%

Failure breakdown (+/-3cm, 38 failed episodes):
  failed_to_lift        21   reached/grasped near the cube but never achieved sustained lift
  pushed_cube_away      13   cube displaced without being lifted
  grasped_but_dropped    3   lifted briefly, lost the grasp
  timeout_uncategorized  1

Language variants (+/-3cm, seed 7, 10 episodes each, 40 total):
  "Pick up the red cube."   1/10  10%
  "Grasp the red cube."     4/10  40%
  "Lift the red cube."      2/10  20%
  "Pick up the red block."  1/10  10%
  (small per-variant samples -- not strong evidence of a real language effect,
   but performance is not uniform across paraphrases either)

Inference latency (closed loop, +/-3cm): mean 10.4ms  p50 9.7ms  p95 15.8ms
Effective control rate: ~96 Hz
```

**Offline vs. closed-loop, side by side:**

```text
                    Offline (Step 4)      Closed-loop (Step 5, +/-3cm)
Joint MAE           0.0029 rad             n/a (no per-step GT in closed loop)
Gripper accuracy    99.95%                 n/a (see failure breakdown instead)
Task success        n/a (BC doesn't        24.0% (12/50), physical sustained lift
                     have a success
                     notion)
```

The two metrics are not directly comparable (offline MAE never had a
chance to compound; closed-loop success is a different kind of measurement
entirely) -- the point is that a policy which reproduces expert actions
almost perfectly one step at a time does not automatically chain those
steps into a working trajectory.

### Failure diagnosis

Inspecting individual rollouts (`evaluation/diagnostics.py`,
`run_vla_demo.py --save-frames`) shows the dominant failure mode is
**oscillation near the cube, not an inability to reach it**: in a
representative failed episode, the end-effector reached within 0.9cm of
the cube by tick 63, but the predicted gripper probability then
oscillated between fully-open and fully-closed several times over the
remaining ~290 ticks (e.g. closed at tick 60, open by tick 100, closed
again by tick 200, open again by tick 300) instead of committing to
close-then-lift. This matches classic behavior-cloning compounding error:
the model was trained on trajectories where "close the gripper" appears
at one consistent phase of a monotonically-progressing expert rollout: in
closed loop, small deviations from that expert trajectory put the policy
in an ambiguous state between "approaching" and "grasping" that it has
never had to resolve, and it flips back and forth instead of committing.

**Minimal remedy tested (opt-in, off by default):** EMA action smoothing
(`--smoothing-alpha`, `evaluation/closed_loop.py::_smooth_action`), motivated
directly by the oscillation evidence above. Raw baseline is always the
default and is what's reported unless stated otherwise.

```text
Raw            (+/-3cm, seed 42): 12/50  24.0%
Smoothed a=0.3 (+/-3cm, seed 42):  0/50   0.0%  -- WORSE, not better
  failure breakdown: gripper_never_closed x50
```

**Smoothing made things categorically worse**, not better: with EMA
weight 0.3 applied to the gripper channel, a smoothed value starting near
1.0 (open) only moves 30% of the way toward a new raw prediction each
tick; since the raw gripper prediction itself flips back toward "open"
again after only a few ticks (the oscillation described above), the
smoothed value never had time to cross the 0.5 closed threshold in any of
the 50 episodes. This is a genuinely useful negative result: naive EMA
smoothing is the wrong fix for a fundamentally binary, fast-switching
signal like gripper state, even though it might plausibly help the
continuous joint channels. No further smoothing configurations were swept
-- the raw (unsmoothed) baseline remains the one reported as "the Step 5
result." No dataset regeneration, retraining, architecture change, or
control-rate change was attempted -- per README policy, that would require
reporting the baseline failed and why first, which this section does.

These are all measured values from actual runs
(`outputs/evaluation/dense_vla_closed_loop/*/summary.json`), not
projections -- re-run the commands below to reproduce them.

## Step 6 -- Sparse Mixture-of-Experts VLA

### Architecture

Only the Transformer FFN sublayers change. Attention, fusion, encoders,
and the action head are byte-for-byte the same classes as Dense.

```text
num_experts   = 4
top_k         = 1  (each token routes to exactly 1 of 4 experts)
moe_layers    = (1, 3)   -- layers 0 and 2 stay dense; alternating hybrid
expert FFN    = Linear(256,1024) -> GELU -> Dropout -> Linear(1024,256) -> Dropout
                (identical shape to the Dense FFN it replaces)
router        = Linear(256, 4) -> softmax -> top-1
```

Routing is **token-level**: the same 4 semantic tokens
(`VISION, LANGUAGE, STATE, ACTION_QUERY`) from Step 4's fusion each get
routed independently at every MoE layer, and the router sees only hidden
states -- no token-identity shortcut is hardcoded (`models/moe.py`,
`models/moe_transformer.py`).

**Top-1 output is unweighted** (`selected_expert(x)`, not scaled by the
router's softmax probability) -- see `models/moe.py`'s docstring for why:
it's what makes a freshly-converted Dense->MoE model reproduce Dense's
output exactly, and it's not needed for router trainability because the
router gets its gradient from the load-balancing auxiliary loss instead
(Switch-Transformer-style: `L_router = num_experts * sum_e(fraction_e * mean_prob_e)`,
`router_aux_loss_weight = 0.01`).

### Dense->MoE initialization

`models/moe_vla.py::convert_dense_to_moe()`: every shared component
(encoders, fusion, action head, attention, both LayerNorms per layer) is
copied verbatim from the trained Dense checkpoint; for each MoE layer,
the Dense FFN's weights are copied into **every** expert (so all 4 start
as exact copies of each other and of Dense); only the router is randomly
initialized. Measured immediately after conversion, before any MoE
training: **max joint-output diff = 2.4e-7, max gripper-logit diff =
4.8e-7** (`outputs/training/moe_vla_run_001/dense_moe_initial_similarity.json`)
-- functionally identical to Dense, as designed
(`tests/test_moe_shapes.py::test_dense_to_moe_conversion_is_functionally_equivalent`).
This is reported as **Dense-initialized MoE**, not MoE trained from
scratch.

### Training

Same dataset, split, normalization (verified numerically identical to
Dense's checkpointed stats), loss (+ router aux term), optimizer (AdamW),
LR (1e-4), epochs (30), seed (42), and device (MPS) as Dense -- see
README "Experimental fairness". Tiny-overfit gate (64 samples, 150
epochs, mandatory before full training): joint loss 0.0531 -> 0.0025
(95% reduction), gripper accuracy 100%.

```bash
python -m training.train_moe --data data/demonstrations \
    --dense-checkpoint outputs/training/dense_vla_run_001/best.pt \
    --epochs 30 --batch-size 32 --learning-rate 1e-4 \
    --num-experts 4 --top-k 1 --moe-layers 1 3 --router-aux-weight 0.01 --seed 42
```

### Parameters

```text
                        Dense           MoE
Total parameters        81,198,408      84,353,872
Trainable parameters     3,659,016       6,814,480
Active params/token     81,198,408      81,200,464
```

"Active parameters per token" = every parameter that actually participates
in computing that token's output: all non-MoE parameters, plus (per MoE
layer) only the 1 expert actually selected -- not all 4 that merely
occupy memory (`models/moe_vla.py::parameter_accounting`). MoE holds
~3.16M extra parameters in memory (3 unused expert copies x 2 MoE layers)
but each token's forward pass touches almost exactly the same parameter
count as Dense (+2K for the two router linear layers).

### Offline results (held-out test split, 2,159 samples)

```text
                     Dense       MoE
Joint MAE (rad)      0.0029      0.0026
Gripper accuracy     99.95%      100.00%
```

MoE offline accuracy is comparable to Dense, marginally better on both
metrics. This alone would suggest MoE gained useful capacity without
hurting imitation quality -- the closed-loop numbers below tell a
different story.

### Expert utilization and specialization (validation set, 2,140 samples)

Real per-token-type routing distributions -- not hardcoded, not cherry-picked:

```text
Layer 1 (closer to input):
  VISION:       E0  4%   E1 96%   E2  0%   E3  0%
  LANGUAGE:     E0  0%   E1  0%   E2  0%   E3 100%
  STATE:        E0  9%   E1  1%   E2 89%   E3  1%
  ACTION_QUERY: E0 85%   E1  2%   E2 13%   E3  0%

Layer 3 (closer to output):
  VISION:       E0  7%   E1 62%   E2 27%   E3  3%
  LANGUAGE:     E0 79%   E1  6%   E2  9%   E3  6%
  STATE:        E0 16%   E1 37%   E2 33%   E3 14%
  ACTION_QUERY: E0 24%   E1 20%   E2 20%   E3 36%
```

**Layer 1 shows strong, clean per-token-type specialization** -- LANGUAGE
routes to Expert 3 100% of the time, VISION to Expert 1 96% of the time,
STATE to Expert 2 89% of the time, each token type landing on a
different, mostly-dedicated expert. **Layer 3's specialization is much
weaker/noisier**, especially for ACTION_QUERY (spread fairly evenly
across all 4 experts). Router entropy confirms this quantitatively:
layer 1 entropy 1.36 nats, layer 3 entropy 1.38 nats (both well below the
uniform max of `ln(4)=1.39`, i.e. routing is non-uniform/informative at
both layers, but the per-token-type breakdown shows layer 1's confidence
is structured while layer 3's is more mixed). No load-balance collapse
occurred (no expert near 0% or 100% of all tokens combined).

Framed carefully (README "Avoid causal overclaiming"): this shows the
router **learned to route different modalities differently** at layer 1
-- a genuine, measured specialization signal -- not that any expert
"understands vision" or "understands language" in a deeper sense.

### Closed-loop results (same protocol as Step 5: same seeds, same cube-offset sequence, `control_substeps=10`, `max_steps=350`, raw/unsmoothed)

```text
                    Dense              MoE
+/-3cm (ID)         12/50  24.0%        0/50   0.0%
+/-4cm (OOD)         9/50  18.0%        1/50   2.0%
+/-5cm (OOD)         6/50  12.0%        2/50   4.0%

Failures +/-3cm:
  Dense: failed_to_lift 21, pushed_cube_away 13, grasped_but_dropped 3, timeout 1
  MoE:   pushed_cube_away 26, failed_to_lift 15, grasped_but_dropped 9

Language variants +/-3cm (seed 7, 10 episodes each):
  Dense: 10% / 40% / 20% / 10%   (combined 8/40 = 20%)
  MoE:    0% /  10% /  0% /  0%   (combined 1/40 = 2.5%)
```

**MoE closed-loop success is worse than Dense at every condition tested**,
despite comparable-or-better offline accuracy. The best MoE episode
(`outputs/evaluation/moe_vla_closed_loop/seed42_xy0.03/episodes.jsonl`)
reached a cube-lift-delta of 0.045m -- just under the 0.045m sustained-lift
threshold -- so this is not a broken/frozen policy (visually confirmed in
`outputs/moe_vla_demo_initial.png` / `moe_vla_demo_final.png`: the arm
reaches for and displaces the cube, same qualitative behavior as Dense),
it is a policy that reaches the grasp region reliably but fails to
convert that into a sustained lift slightly *more* often than Dense does.
MoE's failure mix leans more toward `pushed_cube_away` /
`grasped_but_dropped` (grasp achieved, not held) versus Dense's
`failed_to_lift`-heavy profile -- a shift in *how* it fails, not evidence
that the underlying temporal/compounding-error problem (Step 5's
diagnosis) was solved. It was not: both models still fail primarily
around the grasp-to-lift transition.

### Expert-switching / routing-vs-gripper analysis (15 additional diagnostic-only episodes, +/-3cm)

```text
ACTION_QUERY token mean expert-switch rate (fraction of consecutive
control ticks where the selected expert changed):
  layer 1: 17.1%
  layer 3: 25.7%

Layer 1 gripper-target correlation by selected expert (mean predicted
gripper value while that expert was active; 1=open, 0=closed):
  Expert 0 (85% of ACTION_QUERY traffic offline): mean gripper = 0.08 (closed-leaning)
  Expert 3 (rare):                                 mean gripper = 0.79 (open-leaning)
```

The ACTION_QUERY token's selected expert changes on 17-26% of consecutive
ticks -- far from stable, but also far from constant thrashing. There is
a real, measured association between which expert layer 1 picks and the
gripper value the model outputs at that tick; phrased carefully, this
means **routing correlates with (rather than causes, and rather than
having been designed to reflect) the model's gripper decision** -- a
descriptive finding, not evidence the router discovered a
"grasp-phase expert" in any deeper sense.

### Systems: MoE is slower on this device, not faster

```text
Batch-1 offline latency (identical model/preprocessing, MPS, Apple M5 Max):
  Dense: mean 6.25ms  p50 6.29ms
  MoE:   mean 10.41ms p50 10.42ms   (~1.7x slower)

Closed-loop mean inference latency (+/-3cm):
  Dense: 10.4ms (~96 Hz effective)
  MoE:   13.6ms (~73 Hz effective)
```

Theoretical active compute per token is nearly identical to Dense (see
"Parameters" above: 81.2M active vs 81.2M for Dense). Measured latency is
**not** lower -- on MPS, this implementation's sparse dispatch (looping
over experts and boolean-masking token subsets, see `models/moe.py`'s
`MoEFFN.forward`) adds real overhead that outweighs the FLOP savings from
only running 1-of-4 experts. This is reported as a genuine systems
result, not a claim of MoE being faster: theoretical sparsity and
measured wall-clock speed are two different things, and this codebase
does not claim the latter improved.

### Interpretation

This experiment's answer to "what changes when Dense FFNs become sparse
MoE FFNs": **capacity goes up, real per-token-type routing specialization
emerges (particularly at layer 1), offline imitation quality is
essentially unchanged, and closed-loop task success gets worse, not
better, while inference gets slower on this hardware.** The Step 5
diagnosis -- that Dense's dominant failure mode is a temporal/
compounding-error problem (gripper-timing oscillation), not a capacity or
representation problem -- is corroborated, not contradicted: giving the
model more capacity via sparse experts did not fix a failure mode that
was never about capacity in the first place. This is exactly the kind of
outcome README "Expected scientific possibility" anticipated as plausible,
and it directly motivates a future temporal-history milestone rather than
a bigger/sparser single-timestep model.

### Run it yourself

```bash
python -m training.train_moe --dense-checkpoint outputs/training/dense_vla_run_001/best.pt \
    --overfit-samples 64 --epochs 300              # tiny-overfit gate (run first)

python -m training.train_moe --data data/demonstrations \
    --dense-checkpoint outputs/training/dense_vla_run_001/best.pt \
    --epochs 30 --batch-size 32 --learning-rate 1e-4 --seed 42   # full training

python -m training.evaluate_moe --checkpoint outputs/training/moe_vla_run_001/best.pt \
    --data data/demonstrations --split test          # held-out offline metrics

python -m simulation.run_moe_vla_demo --checkpoint outputs/training/moe_vla_run_001/best.pt

python -m simulation.evaluate_moe_vla_closed_loop \
    --checkpoint outputs/training/moe_vla_run_001/best.pt \
    --episodes 50 --xy-randomization 0.03 --seed 42 --routing-episodes 15

python -m evaluation.compare   # writes outputs/evaluation/dense_vs_moe_vs_temporal_summary.json
```

## Step 7 -- Temporal Dense VLA with Observation/Action History

### Architecture

Same encoders and dense Transformer size as Step 4; the change is purely
in what's fed in and how it's tokenized.

```text
history_length = 4

For each of the 4 window positions (t-3, t-2, t-1, t):
  RGB   -> VisionEncoder (shared, frozen ResNet18)   --\
  State -> StateEncoder (shared)                        +--> sum --> +temporal position embedding --> token
  PrevAction -> ActionHistoryEncoder (shared, new)   --/
  (position t's PrevAction is always a NO_ACTION sentinel -- see below)

[token_t-3, token_t-2, token_t-1, token_t, LANGUAGE, ACTION_QUERY]
    -> Dense Transformer (hidden=256, layers=4, heads=8, ffn=1024 -- same as Dense, NO MoE)
    -> ACTION_QUERY output -> ActionHead -> 8D action
```

One fused token per timestep (README "alternative simpler token layout"),
not one token per (modality, timestep) pair -- keeps the sequence short
(6 tokens for H=4) and the architecture easy to reason about and test.
`models/moe_transformer.py::HybridTransformer` is reused with
`moe_layers=()` for the backbone -- a plain dense Transformer, not MoE,
via the same tested code Step 6 built.

### Temporal sample definition and padding contract

The single source of truth is `models/temporal_history.py`, imported by
both the training dataset builder and the runtime policy so they can
never silently diverge:

```text
For a sample/window ending at timestep t (target = action_t):
  window covers observation indices [t-3, t-2, t-1, t]
  indices < 0 (before episode start) -> left-pad by repeating index 0

  Each slot's PREVIOUS-ACTION value:
    real recorded/issued action at that index,  IF index >= 0 AND slot != last
    NO_ACTION_VECTOR (zeros(7) normalized-joint + gripper=0.5)  OTHERWISE
      -- covers BOTH left-padding (index < 0) AND the current/last slot
         (using the real action there would be target leakage)
```

`Action_t` (the prediction target) never appears anywhere in the model's
input -- enforced by construction (only indices `<= t-1` are ever read
for actions) and checked exhaustively by
`tests/test_temporal_dataset.py::test_target_action_never_appears_in_previous_actions`
plus dedicated cross-episode-leakage tests. History windows never cross
episode boundaries -- each episode's own arrays are indexed independently.

### Previous-action representation

8D: 7 joint targets (normalized with the same train-split
`ActionNormalizer` Dense/MoE use) + 1 raw gripper value in `[0,1]`.
Encoded by a new `ActionHistoryEncoder` (`8 -> 128 -> 256`, GELU +
LayerNorm, same style as `StateEncoder`). No handwritten gripper
persistence rule was added (README "No handwritten gripper rule") --
whatever consistency the model shows, it learned from the data.

### Dense initialization

`models/temporal_vla.py::convert_dense_to_temporal()`: copied verbatim
from the trained Dense checkpoint -- vision/language/state encoders,
action head, `vision_proj`/`state_proj`/`language_proj` (Dense's fusion
projections, reused per-timestep with shared weights), `action_query`,
and every Transformer layer's self-attention + LayerNorms + dense FFN.
Randomly initialized (no Dense equivalent): `action_history_encoder`,
`action_proj`, `temporal_position_embedding`, `extra_token_embedding`.
Unlike Step 6's Dense->MoE conversion, this does **not** reproduce Dense's
output numerically -- the token sequence is structurally different
(temporal window + language + action-query vs. Dense's single-timestep 4
tokens) -- so no functional-equivalence claim is made or tested here.

### Parameters

```text
                        Dense           Temporal
Total parameters        81,198,408      81,299,656   (+0.12%)
Trainable parameters     3,659,016       3,760,264
```

Deliberately close to Dense (README "keep the experiment about history,
not a bigger model") -- the entire increase is the new
`action_history_encoder` + `action_proj` + embeddings.

### Training

Same dataset/split/normalization/loss/optimizer/LR/seed/device as Dense
(AdamW, lr=1e-4, 30 epochs, seed=42, MPS) -- only the architecture and
the `--history-length 4` input differ. Tiny-overfit gate (64 samples, 150
epochs, mandatory before full training): joint loss 0.209 -> 0.0013 (99%
reduction), gripper accuracy 100%.

```bash
python -m training.train_temporal --dense-checkpoint outputs/training/dense_vla_run_001/best.pt \
    --overfit-samples 64 --epochs 300   # tiny-overfit gate, run first

python -m training.train_temporal --data data/demonstrations \
    --dense-checkpoint outputs/training/dense_vla_run_001/best.pt \
    --epochs 30 --batch-size 16 --learning-rate 1e-4 --history-length 4 --seed 42
```

### Offline results (held-out test split, 2,159 samples)

```text
                     Dense       MoE         Temporal
Joint MAE (rad)      0.0029      0.0026      0.0025
Gripper accuracy     99.95%      100.00%     100.00%
```

Comparable to (marginally better than) both prior systems -- as with
Step 6, offline accuracy alone does not predict closed-loop behavior;
see below.

### Runtime: `TemporalDenseVLAPolicy`

```python
policy = TemporalDenseVLAPolicy.from_checkpoint(checkpoint_path, device=device)
policy.reset()                                    # MANDATORY at the start of every episode
action = policy.predict(observation, instruction)  # same simple API as Dense/MoE
```

Internally maintains a rolling buffer of up to `history_length` past
`Observation`s and this policy's OWN previously issued actions (never
external/expert actions), applying the exact same left-pad/mask
convention as training. **Teacher-forcing distinction (README, important):**
during training, the previous-action window is the EXPERT's recorded
actions; at inference, it is the model's own outputs. This is a real,
acknowledged source of train/inference distribution shift, not hidden.
`evaluation/closed_loop.py::run_closed_loop_episode` now calls
`policy.reset()` automatically (a no-op for Dense/MoE, which have no such
method) -- verified never to leak across episodes by
`tests/test_temporal_no_privileged_inputs.py::test_closed_loop_episode_resets_policy_history`.

### Single demo (no randomization)

`outputs/temporal_vla_demo_initial.png` / `_final.png` (sent above):
termination=timeout, cube lift delta 0.013m (not a success this specific
deterministic run), but **only 1 gripper open<->closed switch across all
350 ticks** -- qualitatively different from Dense's frequent flipping in
the equivalent Step 5 demo.

### Closed-loop results (identical protocol/seeds/cube-offsets to Dense/MoE)

```text
                    Dense              MoE                Temporal
+/-3cm (ID)         12/50  24.0%        0/50   0.0%        11/50  22.0%
+/-4cm (OOD)         9/50  18.0%        1/50   2.0%         8/50  16.0%
+/-5cm (OOD)         6/50  12.0%        2/50   4.0%         7/50  14.0%

Language variants +/-3cm (seed 7, 10 episodes each, 40 total):
  Dense:    10% / 40% / 20% / 10%   (combined  8/40 = 20.0%)
  MoE:       0% / 10% /  0% /  0%   (combined  1/40 =  2.5%)
  Temporal: 10% / 20% / 30% / 50%   (combined 11/40 = 27.5%)

Failures +/-3cm:
  Dense:    failed_to_lift 21, pushed_cube_away 13, grasped_but_dropped 3, timeout 1
  MoE:      pushed_cube_away 26, failed_to_lift 15, grasped_but_dropped 9
  Temporal: failed_to_lift 22, pushed_cube_away 15, grasped_but_dropped 2
```

Temporal's success rate tracks Dense closely (within noise of 50-episode
sampling) at every condition, and clearly beats MoE at all three. It does
**not** show a large, clean improvement over Dense's success rate despite
the oscillation fix below -- see "Interpretation".

### Gripper-oscillation comparison (the central Step 7 measurement)

```text
Mean / median gripper open<->closed switches per episode, +/-3cm, 50 episodes:
  Dense:     mean 25.1   median 23.5   max 60
  MoE:       mean 37.9   median 39.0   max 54   (worse than Dense -- Step 6 finding, confirmed)
  Temporal:  mean  2.4   median  1.0   max 17   (~10x fewer than Dense)
```

This is the clearest, most direct evidence in the project so far that the
Step 5/6 diagnosis was correct AND that the specific remedy (short
observation/action history) addresses it: giving the model explicit
control history collapses the oscillation almost entirely. A "grasped but
dropped" outcome fell from 9 (MoE) / 3 (Dense) to 2 (Temporal) -- the
fewest of all three -- consistent with more decisive, persistent gripper
commands once closed.

### Interpretation

**Supported**: short-term temporal context (recent observations + the
policy's own recent actions) dramatically reduces gripper-timing
oscillation -- the specific, concretely-measured failure mode Steps 5-6
diagnosed. This is a real, mechanistic fix, not a wash: median switches
per episode dropped from 23.5 (Dense) to 1 (Temporal).

**Not supported**: that fixing oscillation alone would proportionally
convert Step 5/6's good offline imitation into much better closed-loop
success. Temporal's success rate (22%/16%/14%) is statistically
indistinguishable from Dense's (24%/18%/12%) at n=50 per condition. Gripper
decisiveness improved; overall manipulation success did not follow
proportionally. The remaining failures (`failed_to_lift` 22,
`pushed_cube_away` 15 at ±3cm -- a very similar profile to Dense's) point
to compounding error elsewhere in the trajectory -- e.g. reach/approach
precision, or exposure-bias from teacher-forced training vs. self-fed
inference history (README "Teacher-forcing distinction") -- not solely
gripper timing.

**Systems**: batch-1 offline latency (6.34ms) is essentially unchanged
from Dense (6.25ms) -- MPS batches the 4 history frames through the
shared ResNet efficiently rather than paying a 4x cost. Closed-loop
latency (~12.5ms mean) is higher than Dense's ~10.4ms, most plausibly
from unbatched per-frame Python preprocessing (4 separate PIL transforms)
in the policy's history buffer rather than the model itself -- an
optimization opportunity, not attempted here (README "Optional
frame-encoding optimization" -- not implemented this milestone; the
runtime always re-encodes from raw RGB, prioritizing correctness).

### Dense vs MoE vs Temporal summary

Full table: `outputs/evaluation/dense_vs_moe_vs_temporal_summary.json`
(via `python -m evaluation.compare`).

### Run it yourself

```bash
python -m training.evaluate_temporal --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
    --data data/demonstrations --split test

python -m simulation.run_temporal_vla_demo --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt

python -m simulation.evaluate_temporal_vla_closed_loop \
    --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
    --episodes 50 --xy-randomization 0.03 --seed 42

python -m evaluation.compare
```

## Step 8 -- DAgger / Corrective On-Policy Data

### Core research question

> Does exposing the Temporal VLA to its own off-expert states, labeled
> with corrective expert actions, reduce closed-loop exposure bias and
> improve manipulation success?

Controlled comparison: **Temporal Dense VLA vs. Temporal Dense VLA +
DAgger corrective data**. Architecture, normalization, loss, and the Step
7 checkpoint are held fixed -- the only experimental variable is the
training data. No MoE, action chunking, or RL anywhere in this milestone.

### Corrective expert design (`dagger/corrective_expert.py`)

The Step 2.5 `ScriptedController` is unsuitable for labeling arbitrary
model-visited states: its `stage` field only ever advances forward under
the assumption that IT generated every prior action, so replaying it
against Temporal's own (possibly off-expert) trajectory would desync
almost immediately. `dagger/corrective_expert.py::infer_phase()` instead
re-derives a phase from the CURRENT physical state on every call --
nothing persists across ticks:

```text
gripper closed AND well above the grasp height       -> LIFT (keep lifting)
gripper closed AND positioned AND still at grasp ht.  -> LIFT (settle into it)
gripper closed BUT far from the cube                  -> ABOVE_CUBE (reopen, re-approach)
gripper open AND far from the cube (xy)                -> ABOVE_CUBE
gripper open AND aligned but still high                -> DESCEND
gripper open AND aligned AND low                        -> CLOSE_GRIPPER
```

Deliberately does **not** use live cube-height gain to detect "grasp
secured" (a momentary finger/cube contact bump can lift the cube a few
millimeters before the fingers have actually closed around it -- the
same false-positive `control.success.sustained_lift_success` guards
against for task-success measurement, verified empirically to cause
CLOSE_GRIPPER<->LIFT<->DESCEND phase-flapping and complete grasp failure
when first tried). The gripper-closed threshold (0.022m per-finger
opening) was calibrated to the fingers' *settled* closed-on-cube value
(~0.020-0.021m measured from the Step 3 dataset), not the ~0.030m the
opening passes through mid-close -- using the looser threshold triggered
premature lift attempts before real grip force built up.

Uses the same pose-IK math as `ScriptedController`
(`control.kinematics.solve_pose_ik`, top-down grasp orientation) and is
allowed the same privileged cube position for labeling -- this is teacher
supervision, generating a target only, never a runtime controller (README
"Teacher label generation only"). Validated (`tests/test_corrective_expert.py`):
finite/sensible output on 6 classes of perturbed states (off-trajectory,
beside the cube, too low, partly-closed gripper, post-contact, far
off-workspace) and full closed-loop task success driving itself from HOME
and from a representative ±3cm cube offset.

### Critical runtime distinction

During collection, the corrective expert **only produces a label** --
`dagger/collector.py::collect_episode()` always executes
`policy.predict()`'s action via `env.step()`; `compute_corrective_action()`
is called every tick purely for diagnostics/storage and its return value
never reaches the simulator. Verified directly by
`tests/test_dagger_expert_not_executed.py` (spies on both `env.step` and
the corrective expert, asserts the executed action is always the exact
object `policy.predict()` returned). At final evaluation, no
scripted/corrective expert module is even imported --
`tests/test_dagger_no_privileged_runtime.py`.

### DAgger sample semantics

Each retained sample is:

```text
obs_{t-3:t}  (model-observed RGB/state, left-padded like Step 7)
+ model-issued previous actions_{t-3:t-1}  (NEVER expert actions)
+ language
-> expert corrective action_t  (the target)
```

On disk, a DAgger episode reuses the Step 3 episode format VERBATIM
(`dataset.recorder.EpisodeRecorder` / `dataset.episode.load_episode`) --
`states`/`joint_targets`/`gripper_targets` are the MODEL's own observed
states and executed actions, exactly like a Step 3 episode's are the
expert's. A sibling `expert_labels.npz` holds the corrective label,
per-tick disagreement diagnostics, and a `retained` boolean per timestep.
`dagger/dataset.py::TemporalDaggerCorrectiveDataset` then builds windows
with the identical `models/temporal_history.py` contract Step 7 uses,
substituting the episode's own `joint_targets`/`gripper_targets` for the
history window and `expert_labels.npz`'s value at `t` for the target --
so no future/target/cross-episode leakage is possible by construction
(same guarantee as Step 7, re-verified for DAgger by
`tests/test_dagger_dataset.py` and `tests/test_dagger_alignment.py`).

### Collection (Round 1)

```bash
python -m simulation.collect_dagger_data \
    --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
    --episodes 50 --xy-randomization 0.03 --seed 123 --sample-every 3 \
    --output data/dagger/round_001
```

Seed 123 (distinct from the official evaluation seed 42), ID (±3cm)
distribution only (README "ask first: can DAgger improve the ID gap,
before OOD collection"). Retention: every 3rd tick (periodic backbone) OR
gripper-decision disagreement OR joint-L2 disagreement > 0.15 rad.

```text
Episodes collected: 50   (collection success rate 20.0% -- close to
                           Temporal's own 22% ID rate, as expected: same
                           checkpoint/distribution)
Candidate timesteps: 15,278
Retained DAgger samples: 7,963  (52.1% of candidates)
Mean joint-L2 disagreement (model vs. expert): 0.110 rad
Mean gripper disagreement rate: 15.2% of ticks
```

Retained-sample phase mix (from the corrective expert's own inferred
phase, diagnostic only): 59.1% ABOVE_CUBE, 36.6% LIFT, 2.5% DESCEND, 1.8%
CLOSE_GRIPPER -- see "Interpretation" for why the LIFT share matters.

### Dataset aggregation and fine-tuning

```text
D_train = D_expert (Step 3 train split, 17,144 samples)
          UNION
          D_dagger (45/50 round_001 episodes, 6,933 retained samples;
                     5 episodes / 1,030 samples held out as a
                     corrective-validation split, never trained on)

Batch-sampling mix: 50% expert / 50% DAgger (WeightedRandomSampler,
  dagger/aggregation.py) -- NOT the raw 6,933:17,144 (~29%) ratio.
```

Architecture, normalization (state/action normalizer reused verbatim from
the Step 7 checkpoint, never recomputed), and loss are all unchanged from
Step 7. Fine-tuned (not restarted) from
`temporal_dense_vla_run_001/best.pt`:

```bash
python -m training.train_dagger \
    --base-data data/demonstrations --dagger-data data/dagger/round_001 \
    --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
    --epochs 15 --learning-rate 5e-5 \
    --output outputs/training/temporal_dagger_run_001
```

AdamW, lr=5e-5 (lower than Step 7's 1e-4 fine-tuning convention),
weight_decay=0.01, seed=42, MPS. Best checkpoint selection still uses the
ORIGINAL expert val split's joint MAE (never the DAgger data), so
Temporal and Temporal+DAgger stay apples-to-apples comparable; best.pt =
epoch 14/15 (val_joint_mae 0.00536).

### Offline results

```text
                                    Before fine-tuning   After fine-tuning
Expert test joint MAE (rad)        0.0025               0.0047
Expert test gripper accuracy       100.00%              99.86%
DAgger corrective-val joint MAE    0.0563               0.0213   (-62%)
DAgger corrective-val gripper acc  45.7%                74.2%
```

The corrective-state metric (README "Corrective State Action Error")
improved substantially and in the expected direction -- proof the DAgger
samples reached the training pipeline and were learned (also confirmed
directly by the tiny corrective-overfit sanity check,
`tests/test_dagger_training.py`, before this real run: >50% loss
reduction on held-out-disagreement synthetic samples in 150 epochs).
Original expert-imitation accuracy regressed mildly (joint MAE nearly
doubled, gripper accuracy -0.14pp) -- some forgetting, not catastrophic.
Held-out expert TEST split, never DAgger-contaminated
(`outputs/evaluation/dagger_test_metrics.json`).

### Closed-loop results (identical protocol/seeds/cube-offsets to Dense/MoE/Temporal)

```text
                     Dense       MoE        Temporal    Temporal+DAgger
+/-3cm (ID)          24.0%        0.0%       22.0%        4.0%   (2/50)
+/-4cm (OOD)         18.0%        2.0%       16.0%        2.0%   (1/50)
+/-5cm (OOD)         12.0%        4.0%       14.0%        2.0%   (1/50)

Failures +/-3cm:
  Temporal:        failed_to_lift 22, pushed_cube_away 15, grasped_but_dropped 2
  Temporal+DAgger:  failed_to_lift 40, pushed_cube_away  5, grasped_but_dropped 3

Mean cube lift delta +/-3cm (all episodes): Temporal 0.0191m -> Temporal+DAgger 0.0067m
Gripper switches +/-3cm:  Temporal mean 2.4 / median 1.0   Temporal+DAgger mean 3.0 / median 3.0 / max 9
Batch-1 offline latency:  Temporal 6.34ms -> Temporal+DAgger 6.20ms  (architecture unchanged, as expected)
Closed-loop latency:      ~12.6-12.9ms mean, ~78-80 Hz effective control rate at all three distances
```

**Main success metric, stated plainly: DAgger fine-tuning DECREASED
closed-loop task success at every distance tested** (4.0/2.0/2.0% vs.
Temporal's 22/16/14%). Gripper oscillation was NOT reintroduced (mean 3.0
switches, nowhere near MoE's 37.9) -- the regression is specific to task
completion, not a return of the Step 5/6 oscillation failure mode.

### Root-cause diagnosis (why success collapsed despite better offline corrective-state accuracy)

A single-episode trajectory trace (device=cpu, unperturbed cube,
`record_trajectory=True`) makes the failure mode concrete:

```text
tick   0: eef-cube distance 0.51m, gripper open      (episode start)
tick  40: eef-cube distance 0.11m, gripper open      (approach -- looks right)
tick  80: eef-cube distance 0.10m, gripper CLOSED     (closing near the cube)
tick 120: eef-cube distance 0.19m, gripper closed, z risen 0.52->0.61
tick 349: eef-cube distance 0.19m, gripper closed, IDENTICAL pose as tick ~120
Cube height gain across the ENTIRE 350-tick episode: ~0.00m (never moved)
```

Approach is fine. The gripper closes near the cube -- but the cube never
actually leaves the table (no real grasp was ever secured). The policy
then executes a "lift" retraction anyway, ending up hovering at a fixed,
wrong pose for the remaining ~230 ticks with **no attempt to recover or
re-approach**. This is a direct, learned consequence of the corrective
expert's own design gap identified above: `infer_phase()`'s LIFT trigger
fires from gripper-closed + position alone, with **no verification that
the cube is actually being carried** -- exactly the same simplification
`ScriptedController` gets away with via its fixed 70-tick
close-and-settle hold, which the corrective labeler has no equivalent
for. Because 36.6% of the retained DAgger corrective data carries this
LIFT label -- generated overwhelmingly from Temporal's OWN failed
attempts (78% of the 50 collection episodes did not succeed) -- the model
was taught, at scale, to commit confidently to a lift/retract motion
immediately after closing the gripper, regardless of whether anything was
actually grasped. The failure-taxonomy shift is consistent with exactly
this: `pushed_cube_away` fell (15 -> 5, i.e. the model got MORE cautious/
precise on approach, arguably a genuine improvement) while
`failed_to_lift` rose sharply (22 -> 40) and mean cube-lift-delta
collapsed 3x (0.019m -> 0.007m) -- the model isn't crashing into the
cube anymore, it's giving up on securing it.

### Interpretation

**Not supported**: corrective on-policy data, as collected and weighted
in this experiment, did **not** reduce exposure bias or improve
closed-loop manipulation -- it substantially hurt it. The corrective-state
offline metric improved (62% joint-MAE reduction on exactly the states it
was trained to fix) while closed-loop success fell by more than 4x,
underscoring this project's recurring finding (Steps 5-7) that offline
metrics do not predict closed-loop behavior.

**Diagnosed cause**: the corrective expert's LIFT-phase label does not
verify an actual secured grasp before committing (README "Important
expert-state problem" / "Unrecoverable states should be identified and
optionally excluded" -- this one was not, and should have been). Because
most of Temporal's own collection-time rollouts fail, and failed rollouts
spend most of their length in states the labeler classifies as LIFT, this
flawed label is heavily represented in the retained corrective data and
was learned at scale, producing a MORE consistent "close then abandon"
behavior than Temporal's original (imperfect but more varied) policy.

**Partially supported**: the model DID become measurably more cautious
about one specific failure mode (`pushed_cube_away` fell 15->5, cube-
disturbing collisions are rarer) -- a real, if small, behavioral change
in the intended direction, just outweighed by the new failure mode it
introduced.

**Not the explanation**: gripper oscillation (README's other Step 7
concern) was not reintroduced (mean switches 3.0, essentially unchanged
from Temporal's 2.4) -- this regression is specific to the LIFT-commitment
behavior above, not a general destabilization of the policy.

**Second DAgger round**: not justified from this data. Per README
"Interpretation Rules" (do not tune endlessly until positive), Round 2
should not simply repeat this recipe -- see "What should be tested next."

### Comparison summary

Full table: `outputs/evaluation/dense_vs_moe_vs_temporal_vs_dagger_summary.json`
(via `python -m evaluation.compare`).

### What should be tested next

Not attempted this milestone (README "do not tune endlessly"), but
directly implied by the diagnosis above: (1) add a grasp-verification
gate to the corrective expert's LIFT trigger (e.g. require a brief
sustained cube-height gain, mirroring `control.success`'s own
false-positive guard, before labeling LIFT rather than CLOSE_GRIPPER/
re-approach); (2) cap how many consecutive same-phase corrective samples
one failed episode can contribute, so one long stuck rollout can't
dominate the retained set; (3) re-run Round 1 collection/fine-tuning with
the corrected labeler and re-evaluate before considering a Round 2.

### Run it yourself

```bash
python -m simulation.collect_dagger_data \
    --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
    --episodes 50 --xy-randomization 0.03 --seed 123 --sample-every 3 \
    --output data/dagger/round_001

python -m training.train_dagger --base-data data/demonstrations \
    --dagger-data data/dagger/round_001 \
    --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
    --epochs 15 --learning-rate 5e-5 --output outputs/training/temporal_dagger_run_001

python -m training.evaluate_dagger --checkpoint outputs/training/temporal_dagger_run_001/best.pt \
    --data data/demonstrations --split test --dagger-data data/dagger/round_001

python -m simulation.evaluate_dagger_vla_closed_loop \
    --checkpoint outputs/training/temporal_dagger_run_001/best.pt \
    --episodes 50 --xy-randomization 0.03 --seed 42

python -m evaluation.compare
```

## Step 9 -- ROS2 Deployment + Robot Backend Abstraction

### Core systems question

> Can the existing VLA policy be deployed as a modular ROS2 component
> without depending on MuJoCo internals, while preserving the same
> observation/action contract and measurable closed-loop behavior?

This milestone does not touch model accuracy: no architecture change, no
retraining, all Step 4-8 checkpoints and closed-loop numbers stay frozen.
It converts the research prototype into a layered robotics software
architecture:

```text
Policy            knows ML only
RobotBackend       knows robot execution only
ROS2 bridge        knows transport only
SimulationEnvironment  knows MuJoCo only
```

### Environment (documented, not assumed)

```text
OS:                 macOS (Darwin), Apple Silicon
Python:             3.14 (repo .venv) / ROS2 packages target 3.10-3.12
ROS2 distribution:  NOT INSTALLED in this development environment
                     (no `ros2` CLI, no `rclpy`, no /opt/ros) -- see
                     "Limitations" below. The ROS2 package/nodes/launch
                     file are written correctly against the standard
                     rclpy/ROS2 Jazzy-or-Humble API but have not been
                     built or executed here.
```

### `RobotBackend`: the abstraction (`robot_backend/`)

```python
class RobotBackend(ABC):
    def get_observation(self) -> Observation: ...
    def execute_action(self, action: RobotAction) -> None: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

`MuJoCoBackend` wraps the existing, UNMODIFIED `SimulationEnvironment` --
no physics logic duplicated, just `env.get_observation()`/`env.step()`/
`env.reset()` behind the interface. Privileged evaluation-only calls
(`get_object_position`, Jacobians) are deliberately NOT part of the ABC --
they stay backend-specific passthroughs used only by evaluation code,
never by the policy loop. `FakeRobotBackend` is a MuJoCo-free test double
proving policy code has no MuJoCo dependency
(`tests/test_robot_backend.py`). `FutureHardwareBackend`
(`robot_backend/future_hardware_backend.py`) is a documented,
intentionally-unimplemented extension point (raises `NotImplementedError`
on construction) -- proof the future real-robot contract needs no ABC
changes, not a claim that hardware support exists.

`robot_backend/backend_closed_loop.py::run_closed_loop_episode_via_backend`
is the refactored direct runner (README recommended execution order step
5): the same rollout, driven through `RobotBackend` instead of a raw
`SimulationEnvironment`. Verified to produce **byte-identical** behavior
to the original direct path (same episode length, success, cube
trajectory, gripper probabilities) for the same seed/checkpoint --
`tests/test_backend_closed_loop_equivalence.py`. The original direct path
(`evaluation/closed_loop.py`) is untouched.

### ROS2-adjacent logic with zero `rclpy` dependency (`ros_integration/`)

README "Dependency isolation": the non-ROS2 training/evaluation code must
keep working on a machine without ROS2. Every ROS2-facing DECISION --
serialization, command validation, the watchdog, observation
synchronization, staleness, instruction caching, episode-reset ordering --
lives in plain-Python classes here, fully unit-tested without `rclpy`
installed. The real `rclpy.Node` files under `ros2_ws/` are thin wrappers
that only wire these into topics/services/timers.

```text
ros_integration/
├── serialization.py       Observation/RobotAction <-> ROS-message-shaped field dicts
│                           (sensor_msgs/Image, sensor_msgs/JointState, custom VLARobotAction)
├── command_validator.py   shape / finite / gripper-range / extreme-joint-delta checks
├── watchdog.py             CommandWatchdog: hold-last-safe-action on command timeout
├── sync.py                 LatestMessageSynchronizer + StalenessChecker
├── instruction_cache.py    caches the latest /task_instruction, no per-tick requirement
├── episode_manager.py      backend.reset() -> policy.reset() -> metrics reset, in order
├── policy_node_core.py     VLAPolicyNodeCore: the vla_policy_node's entire control loop
└── bridge_node_core.py     MuJoCoBridgeNodeCore: the mujoco_bridge_node's validate+execute+watchdog loop
```

`VLAPolicyNodeCore.tick(now)` is the whole "VLA Policy Node" control loop
as one testable unit: skip if unsynchronized, skip if stale (log a
warning, never act on old data -- README "Stale data handling"), else
build an `Observation` and call `policy.predict()`. `MuJoCoBridgeNodeCore`
mirrors this for the receiving side: validate every incoming command
before it can become a `RobotAction`, feed the watchdog, and on every
execution tick either run the fresh command or hold the last safe one.

### ROS2 package (`ros2_ws/src/`, written but not built/executed -- see Limitations)

```text
ros2_ws/src/
├── vla_robot_control_msgs/        (ament_cmake -- message/service generation)
│   ├── msg/VLARobotAction.msg      stamp, joint_targets[7], gripper_target
│   └── srv/ResetEpisode.srv        --- success (bool), message (string)
└── vla_robot_control/              (ament_python -- Python/PyTorch runtime)
    ├── vla_robot_control/
    │   ├── mujoco_bridge_node.py    the ONLY ROS2 node that imports MuJoCoBackend
    │   └── vla_policy_node.py       never imports MuJoCo (statically verified)
    ├── launch/mujoco_vla.launch.py
    └── config/default_params.yaml
```

**Nodes**

```text
mujoco_bridge_node   publishes /vla/camera/image, /vla/robot/state (timer, publish_rate_hz)
                      subscribes /vla/action (VLARobotAction)
                      serves /reset_episode (ResetEpisode)
                      the ONLY ROS2-layer code touching SimulationEnvironment

vla_policy_node       subscribes /vla/camera/image, /vla/robot/state, /task_instruction (String)
                       publishes /vla/action (VLARobotAction), on a TIMER at control_frequency_hz
                       (never driven directly by subscriber callback timing)
```

**QoS** (explicit, never left at implicit defaults):

```text
Camera / robot state:  BEST_EFFORT, depth 1   -- an occasional dropped frame must not
                                                  block the newest one; staleness checks handle gaps
Action commands:        RELIABLE, depth 5      -- a dropped command is worse than a dropped frame;
                                                  the watchdog bounds staleness if the publisher stalls
Instruction:             RELIABLE + TRANSIENT_LOCAL, depth 1 -- a (re)started node should see the
                                                  last instruction, not wait for the next change
```

**Runtime parameters** (ROS2 params, never hardcoded paths): `checkpoint`
(required, no default), `policy_type` (`dense`/`moe`/`temporal`/`dagger`
-- `robot_backend/policy_factory.py`), `device`, `instruction`,
`control_frequency_hz`, `camera_topic`/`state_topic`/`action_topic`/
`instruction_topic`, `sync_max_delta_sec`, `stale_timeout_sec`,
`command_timeout_sec`, `max_joint_delta`.

**Launch**:

```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch vla_robot_control mujoco_vla.launch.py \
    checkpoint:=outputs/training/temporal_dense_vla_run_001/best.pt \
    policy_type:=temporal \
    instruction:="Pick up the red cube."
```

`ros2_ws`'s Python packages import this repository's `robot_backend`/
`ros_integration`/`models`/`training`/`control`/`observations`/
`simulation` -- add the repo root to `PYTHONPATH` (or `pip install -e .`
it) before building/running the workspace.

### Reset (`/reset_episode` service, not a topic)

A service (not a topic) so the caller gets a synchronous acknowledgement
that reset actually completed. Ordering (`ros_integration/episode_manager.py`,
tested): **backend.reset() -> policy.reset() -> metrics reset**. Backend
first so any policy inference immediately after sees the post-reset
scene, not a stale one; policy second so `TemporalDenseVLAPolicy`'s
observation/action history never leaks into the next episode (README
"Temporal policy reset" -- verified for the underlying orchestration by
`tests/test_ros_episode_manager.py::test_temporal_policy_history_does_not_leak_across_episode_manager_resets`,
mirroring the Step 7 `run_closed_loop_episode` reset guarantee at the
ROS2-adjacent layer).

### Watchdog behavior

`CommandWatchdog(timeout_sec)`: every backend-execution tick asks "is the
last received command still fresh?" If yes, execute it. If the policy
node stops publishing for longer than `command_timeout_sec`, the backend
**holds the last safe action** rather than doing nothing or silently
re-executing a stale command as if it were current -- a simulation-only
policy, explicitly not a certified safety behavior (see "Avoid
overclaiming" below). `timeout_count` increments once per stale
transition, not once per tick spent stale --
`tests/test_ros_watchdog.py`/`test_ros_bridge_node_core.py`.

### Tests

```text
robot_backend abstraction:        tests/test_robot_backend.py                       (18 tests)
direct-vs-backend equivalence:    tests/test_backend_closed_loop_equivalence.py      (2 tests)
future hardware stub:             tests/test_future_hardware_backend_stub.py         (2 tests)
serialization:                    tests/test_ros_serialization.py                    (11 tests)
command validation:               tests/test_ros_command_validator.py                (12 tests)
watchdog:                         tests/test_ros_watchdog.py                         (8 tests)
observation sync/staleness:       tests/test_ros_sync.py                             (6 tests)
episode reset ordering:           tests/test_ros_episode_manager.py                  (10 tests)
VLA policy node control loop:     tests/test_ros_policy_node_core.py                 (6 tests)
MuJoCo bridge node control loop:  tests/test_ros_bridge_node_core.py                 (8 tests)
real ROS2 node files (rclpy):     tests/test_ros2_node_files.py     -- SKIPPED here, see below
```

All of the above except the last file are plain pytest, no ROS2 required.
`test_ros2_node_files.py` is marked `@pytest.mark.ros2` and gated by
`pytest.importorskip("rclpy")`:

```bash
pytest                    # runs everything; the ros2 file self-skips cleanly (no error)
pytest -m "not ros2"      # explicit pure-Python suite
pytest -m ros2             # ROS2 integration suite -- needs a real ROS2 install; skipped here
```

**pytest: 351 passed, 1 skipped** (272 Step 1-8 baseline + 79 new
pure-Python Step 9 tests; zero regressions in any prior step). The 1
skip is `tests/test_ros2_node_files.py`, cleanly self-skipped via
`pytest.importorskip("rclpy")` -- not an error.

### Measured latency: direct vs. `RobotBackend`-mediated (5 episodes, seed 42, MPS)

```bash
python -m evaluation.backend_benchmark --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt --episodes 5
```

```text
                          Direct (SimulationEnvironment)   Via RobotBackend
Inference latency (ms)    mean 12.92  p50 12.53  p95 14.42   mean 12.73  p50 12.46  p95 15.12
Execute latency (ms)      mean  0.13  p50  0.11  p95  0.26   mean  0.14  p50  0.12  p95  0.25
```

Confirms the `RobotBackend` abstraction itself adds no measurable
overhead (both columns agree within noise) -- the ROS2 transport-latency
column of the full comparison this milestone asks for is NOT included
here; see "Limitations" (`outputs/evaluation/ros2/direct_vs_backend_latency.json`).

### Limitations (README "Avoid overclaiming")

**No ROS2 distribution is installed in this development environment** (no
`ros2` CLI, no `rclpy`). This is a real, load-bearing constraint on what
could actually be verified this milestone:

```text
Verified by execution:      RobotBackend / MuJoCoBackend / FakeRobotBackend,
                             direct-vs-backend equivalence, all
                             ros_integration/ logic (serialization,
                             validation, watchdog, sync, episode reset)

Written, NOT executed:      the ros2_ws/ ROS2 package itself (colcon build),
                             both rclpy nodes, the launch file, the
                             multi-step ROS2 smoke rollout, ANY live
                             latency/throughput/message-drop measurement,
                             the +/-3cm ROS2 closed-loop evaluation
```

The "direct vs ROS2" comparison the milestone asks for
(`outputs/evaluation/ros2/direct_vs_backend_latency.json`) is therefore
**partial and honestly labeled**: it measures direct-`SimulationEnvironment`
vs. `RobotBackend`-mediated latency (real, measured) but NOT ROS2
message-transport latency, throughput, or dropped/stale-message counts
(not measured -- no ROS2 runtime available; fabricating these numbers
would defeat the point of measuring anything). The ROS2 nodes/launch file
are believed correct against the public rclpy API from careful reading,
not proven correct by running them.

This is a **ROS2-based simulated deployment architecture with backend
abstraction and runtime safety checks** -- not a real-time-certified or
production-ready physical-robot safety system, and not yet validated
end-to-end on a real ROS2 install.

### What Step 9 adds, and what's still needed before real hardware

Adds: a real backend abstraction (proven by a MuJoCo-free fake backend
and byte-identical direct-vs-backend equivalence), a correctly-designed
ROS2 node graph with explicit QoS, asynchronous sensor synchronization,
runtime command validation, a watchdog, and a documented (not
implemented) hardware extension point.

Still needed before ANY real robot: build and run the ROS2 workspace on
an actual ROS2 install; execute the multi-step smoke test and the
`+/-3cm` ROS2-vs-direct comparison for real; a genuine hardware driver
(Franka/UR5) implementing `RobotBackend`; a real, non-simulation-only
safety supervisor (Step 10 territory, README "Minimal safety boundary" --
"a larger safety framework belongs to Step 10"); a real task-success
signal (there is no `get_object_position` on physical hardware).

## What's here (Step 4 + Step 5 + Step 6 + Step 7 + Step 8 + Step 9)

- `models/` -- `vision_encoder.py`, `language_encoder.py`,
  `state_encoder.py` (per-modality encoders), `fusion.py`
  (`MultimodalFusion`, the token layout), `dense_vla.py` (`DenseVLA`,
  `DenseVLAConfig`, `count_parameters`), `policy.py` (`DenseVLAPolicy`,
  the inference adapter -- never imports MuJoCo).
- `training/` -- `config.py` (`TrainingConfig`, device/seed helpers),
  `normalization.py` (`StateNormalizer`, `ActionNormalizer`,
  `fit_normalizers_from_split`), `losses.py` (`compute_loss`),
  `checkpoint.py` (`save_checkpoint`/`load_checkpoint`), `train.py`
  (`python -m training.train`), `evaluate.py` (`python -m training.evaluate`,
  also the shared `evaluate_model()` used by validation each epoch).
- `dataset/torch_dataset.py` -- `DemonstrationTorchDataset`, wraps the
  Step 3 `DemonstrationDataset` with image transform + tokenization +
  normalization into training-ready tensors. Read-only w.r.t.
  `data/demonstrations` -- writes nothing back into episode directories.
- `checkpoints/`, `outputs/training/<run>/` -- not committed to git
  (`.gitignore`); each run directory holds `config.json`,
  `training_history.json`, `metrics.json`, `best.pt`, `last.pt`.
- `evaluation/` -- `closed_loop.py` (`run_closed_loop_episode()`, the
  shared rollout both `run_vla_demo.py` and `evaluate_vla_closed_loop.py`
  use; enforces the no-privileged-input boundary), `metrics.py`
  (success-rate/latency aggregation, heuristic `classify_failure()`),
  `diagnostics.py` (optional nearest-training-state drift analysis, not
  run by default).
- `simulation/run_vla_demo.py` -- single closed-loop episode under the
  learned policy, saves before/after (and optionally periodic) frames.
- `simulation/evaluate_vla_closed_loop.py` -- N-episode closed-loop
  evaluation with configurable randomization/instruction/smoothing.
- `outputs/evaluation/dense_vla_closed_loop/<run>/` -- not committed to
  git; `summary.json`, `episodes.jsonl`, `latency.json`, `failure_counts.json`.
- `models/moe.py` -- `FeedForward` (shared dense/expert FFN shape),
  `MoEFFN` (router + real conditional dispatch), `load_balance_loss()`,
  `router_entropy()`.
- `models/moe_transformer.py` -- `TransformerBlock` (dense attention +
  swappable dense-or-MoE FFN), `HybridTransformer`.
- `models/moe_vla.py` -- `MoEVLAConfig`, `MoEVLA`, `convert_dense_to_moe()`,
  `parameter_accounting()`.
- `models/moe_policy.py` -- `MoEVLAPolicy` (mirrors `DenseVLAPolicy`;
  `predict()` for control, `predict_with_routing()` for evaluation-only
  diagnostics).
- `training/train_moe.py` / `evaluate_moe.py` -- MoE training (Dense-init
  conversion, tiny-overfit gate, router-aux loss) and held-out evaluation
  (adds router entropy + expert utilization to Dense's metric set).
- `evaluation/moe_diagnostics.py` -- `run_routing_analysis_episode()`,
  `expert_switch_rate()`, `gripper_expert_correlation()` (all
  evaluation-only, post-hoc; never fed back into the model).
- `evaluation/compare.py` -- `python -m evaluation.compare`, builds
  `outputs/evaluation/dense_vs_moe_vs_temporal_summary.json` (Dense/MoE/
  Temporal columns, any missing system simply omitted).
- `simulation/run_moe_vla_demo.py` / `evaluate_moe_vla_closed_loop.py` --
  MoE analogues of the Step 5 Dense scripts; reuse
  `evaluation.closed_loop.run_closed_loop_episode` unmodified.
- `outputs/training/moe_vla_run_001/` -- adds `router_history.json`,
  `expert_utilization.json`, `dense_moe_initial_similarity.json` to the
  standard training-run outputs; not committed to git.
- `outputs/evaluation/moe_vla_closed_loop/<run>/` -- same layout as Dense's,
  plus `routing_summary.json` when `--routing-episodes` is used.
- `models/temporal_history.py` -- the padding/masking contract shared by
  training and runtime (`NO_ACTION_VECTOR`, `build_action_window()`, etc.).
- `dataset/temporal_torch_dataset.py` -- `TemporalDemonstrationDataset`,
  builds `(history ending at t) -> action_t` windows from the existing
  Step 3 episodes; no new demonstrations generated.
- `models/temporal_vla.py` -- `TemporalDenseVLAConfig`, `TemporalDenseVLA`,
  `ActionHistoryEncoder`, `convert_dense_to_temporal()`. Reuses
  `models.moe_transformer.HybridTransformer` with `moe_layers=()` for the
  (dense, non-MoE) Transformer backbone.
- `models/temporal_policy.py` -- `TemporalDenseVLAPolicy`: rolling
  observation/action history buffer, `reset()`, same simple `predict()`
  API as Dense/MoE.
- `training/train_temporal.py` / `evaluate_temporal.py` -- Dense-init
  conversion, tiny-overfit gate, training, and held-out evaluation.
- `evaluation/temporal_diagnostics.py` -- `gripper_switch_count()`,
  `gripper_switch_rate()`, `summarize_gripper_stability()` -- the central
  Step 7 metric, also computed post-hoc for Dense/MoE by `evaluation/compare.py`.
- `simulation/run_temporal_vla_demo.py` / `evaluate_temporal_vla_closed_loop.py`
  -- Temporal analogues of the Dense/MoE scripts.
- `outputs/training/temporal_dense_vla_run_001/`,
  `outputs/evaluation/temporal_vla_closed_loop/<run>/` -- not committed to git.
- `dagger/corrective_expert.py` -- stateless per-tick phase inference +
  corrective-action labeler (teacher-only; never on the policy execution path).
- `dagger/disagreement.py` -- `compute_disagreement()`, `should_retain()`
  (periodic + disagreement-triggered DAgger sampling policy).
- `dagger/collector.py` -- `collect_episode()`: the ONLY module that drives
  a DAgger rollout; model action executed, expert action labeled/stored only.
- `dagger/dataset.py` -- `TemporalDaggerCorrectiveDataset`: same
  `models/temporal_history.py` window contract as Step 7, model-issued
  action history, expert-labeled target.
- `dagger/aggregation.py` -- `build_aggregated_dataloader()`: expert +
  DAgger data mixed at an explicit, reported sampling ratio (default 50/50).
- `training/train_dagger.py` / `evaluate_dagger.py` -- fine-tunes from a
  Temporal checkpoint on the aggregated dataset; before/after evaluation
  on both the original expert test split and a held-out DAgger
  corrective-validation subset.
- `simulation/collect_dagger_data.py` -- DAgger collection CLI.
- `simulation/run_dagger_vla_demo.py` / `evaluate_dagger_vla_closed_loop.py`
  -- Temporal+DAgger analogues of the Step 7 scripts; reuse
  `TemporalDenseVLAPolicy` and `evaluation.closed_loop` unmodified (same
  runtime API, no expert dependency).
- `evaluation/dagger_diagnostics.py` -- lightweight, post-hoc-only
  exposure-bias (nearest-expert-state distance) and recovery-event
  diagnostics; never affects any action or headline metric.
- `data/dagger/round_001/` -- DAgger episodes (Step-3-format +
  `expert_labels.npz` sidecar) + `manifest.json`; not committed to git.
- `outputs/training/temporal_dagger_run_001/`,
  `outputs/evaluation/temporal_dagger_vla_closed_loop/<run>/` -- not
  committed to git; Step 7's own run directories are untouched.
- `robot_backend/` -- `base.py` (`RobotBackend` ABC), `mujoco_backend.py`
  (`MuJoCoBackend`, wraps `SimulationEnvironment`), `fake_backend.py`
  (`FakeRobotBackend`, MuJoCo-free test double), `backend_closed_loop.py`
  (`run_closed_loop_episode_via_backend`, the refactored direct runner),
  `policy_factory.py` (`build_policy(policy_type, checkpoint, device)`),
  `future_hardware_backend.py` (documented, unimplemented extension point).
- `ros_integration/` -- `rclpy`-free ROS2 logic: `serialization.py`,
  `command_validator.py`, `watchdog.py`, `sync.py`, `instruction_cache.py`,
  `episode_manager.py`, `policy_node_core.py`, `bridge_node_core.py`.
- `ros2_ws/src/vla_robot_control_msgs/` -- `VLARobotAction.msg`,
  `ResetEpisode.srv` (ament_cmake); `ros2_ws/src/vla_robot_control/` --
  `mujoco_bridge_node.py`, `vla_policy_node.py`, `launch/mujoco_vla.launch.py`,
  `config/default_params.yaml` (ament_python). Written but not built/run
  in this environment -- see Step 9 "Limitations".
- `evaluation/backend_benchmark.py` -- direct-vs-`RobotBackend` latency
  comparison; `outputs/evaluation/ros2/` -- Step 9 benchmark outputs, kept
  separate from model-training outputs.
- `pytest.ini` -- registers the `ros2` marker
  (`tests/test_ros2_node_files.py`); everything else is plain pytest.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

**Always run the tiny-subset overfit check first** -- if this doesn't
drive the loss down hard, nothing about full training results can be
trusted (see `tests/test_overfit_tiny_batch.py` for the automated version):

```bash
python -m training.train --overfit-samples 64 --epochs 150 --batch-size 16
```

Full training:

```bash
python -m training.train --data data/demonstrations --epochs 30 \
    --batch-size 32 --learning-rate 1e-4 --seed 42
```

Device is auto-selected (CUDA > MPS > CPU); override with `--device`.
Useful flags: `--num-workers`, `--weight-decay`, `--gripper-loss-weight`,
`--train-vision-encoder`, `--train-language-encoder`, `--resume
<checkpoint.pt>`. Each run writes to a fresh
`outputs/training/dense_vla_run_NNN/`.

## Evaluate (held out)

```bash
python -m training.evaluate --checkpoint outputs/training/dense_vla_run_001/best.pt \
    --data data/demonstrations --split test
```

Reports offline joint MAE (physical units, denormalized), per-joint MAE,
gripper accuracy, 8D action MAE, and a few printed GT-vs-predicted
examples. Test split is evaluated once, after model/hyperparameters are
already chosen from validation -- never used for model selection.

## Closed-loop Dense VLA (Step 5)

Single episode, visibly inspectable (`Controller: Dense VLA Policy` in the
log, not the scripted expert):

```bash
python -m simulation.run_vla_demo --checkpoint outputs/training/dense_vla_run_001/best.pt
```

Multi-episode evaluation:

```bash
python -m simulation.evaluate_vla_closed_loop \
    --checkpoint outputs/training/dense_vla_run_001/best.pt \
    --episodes 50 --xy-randomization 0.03 --seed 42
```

Useful flags: `--instruction` / `--all-instructions` (cycle the 4 training
paraphrases), `--max-steps`, `--save-trajectories`, `--smoothing-alpha`
(opt-in EMA smoothing, off by default -- always measure the raw baseline
first). The policy never receives cube position, Jacobian, or controller
stage -- only `Observation` (RGB + 23D state) and the instruction string.

## Run other demos

```bash
python -m simulation.run_simulation             # Step 1: camera + Observation only
python -m simulation.run_robot_demo              # Step 2.5: one full pose-controlled pick episode
python -m simulation.evaluate_grasp_reliability  # Step 2.5: 10 + 10 trial reliability sweep
python -m dataset.generate_dataset --episodes 10 --seed 42  # Step 3: generate demonstrations
```

## Run tests

```bash
pytest                # everything; the ros2-marked file self-skips cleanly if rclpy is absent
pytest -m "not ros2"  # explicit pure-Python suite (no ROS2 needed)
pytest -m ros2         # ROS2 integration suite only -- needs a real ROS2 (rclpy) install
```

## Platform notes

Tested on macOS (Apple Silicon, MPS backend) with MuJoCo's default OpenGL
backend. If rendering fails in a headless environment, set `MUJOCO_GL=egl`
(Linux) or `MUJOCO_GL=glfw`/`cgl` (macOS). Training auto-selects CUDA if
available, then MPS, then CPU.

Generated datasets (`data/demonstrations/`), training outputs
(`checkpoints/`, `outputs/training/`), and closed-loop evaluation outputs
(`outputs/evaluation/`) are not committed to git -- see `.gitignore`.
Regenerate with the commands above.

## Future roadmap

```text
Step 8 finding to build on (not started):
  - DAgger Round 1, as configured, REGRESSED closed-loop success
    (22%->4% at +/-3cm) -- root cause diagnosed as the corrective expert's
    LIFT-phase label not verifying an actual secured grasp before
    committing (see README "Root-cause diagnosis" / "What should be
    tested next")
  - fix the corrective expert (grasp-verification gate on LIFT, cap
    consecutive same-phase samples per failed episode) and re-run before
    considering any second DAgger round
  - action chunking (predict Action_t:t+k, not just Action_t) -- still
    deliberately deferred, now with an extra reason: a chunked policy
    committing to several steps at once would make a bad LIFT-style
    commitment even harder to recover from until the corrective-expert
    fix above lands
Step 9 finding to build on (not started):
  - the ROS2 package (ros2_ws/) was written correctly against the public
    rclpy API but never built or executed -- no ROS2 distribution is
    installed in this development environment (see Step 9 "Limitations")
  - before anything else in a ROS2-capable environment: colcon build,
    run the multi-step smoke test (tests/test_ros2_node_files.py, `pytest
    -m ros2`), then the real +/-3cm ROS2-vs-direct closed-loop comparison
  - action chunking (predict Action_t:t+k, not just Action_t) -- still
    deliberately deferred, and now also blocked behind the Step 8
    corrective-expert fix (a chunked policy would make a bad LIFT-style
    commitment even harder to recover from)
Step 10 — industrial safety supervisor (README "a larger safety framework
          belongs to Step 10") + real hardware backend implementation,
          once a physical robot is in scope
Step 11 — action chunking (pending the Step 8 fix above)
Step 12 — final benchmark
```
