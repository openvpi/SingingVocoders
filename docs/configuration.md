# Configuration and recipes

## Resolution order

Each YAML file may declare a `bases` list. Base paths are resolved relative to the file that declares them. Values merge in this order:

1. bases from left to right;
2. the current leaf file;
3. repeated CLI `--set key=value` values.

OmegaConf performs composition and interpolation resolution. Pydantic then validates the fully resolved object with unknown outer fields forbidden. Inheritance cycles and unresolved interpolation fail before any model or dataset is created.

```bash
vocoder config validate nsf-hifigan \
  --set data.batch_size=2 \
  --set runtime.precision=bf16-mixed
```

`config validate` also constructs the complete reflective component graph, so invalid constructor parameters and contract violations are detected before training.

## Dynamic validation

The configuration system provides the dynamic behavior used by the GAME and FA2026 frameworks without copying their scope bitmasks or expression-language implementation. The equivalent responsibilities are split across standard tools:

| Configuration responsibility | Implementation |
| --- | --- |
| Inheritance and late overrides | Relative `bases`, leaf merge, then OmegaConf dotlist overrides |
| References to another field | OmegaConf interpolation such as `${data.features.hop_size}` |
| Conditional configuration shapes | Pydantic discriminated unions for objectives, optimization plans, optimizers, and schedulers |
| Cross-field checks | Pydantic model validators for feature limits, model/data dimensions, plan/objective pairing, warm-up boundaries, and optimizer routes |
| Backbone-specific open parameters | `ComponentSpec.params`, followed by exact constructor binding and full component construction |
| Runtime contracts | Required Generator/Discriminator base contracts and typed batch/output validation |

This project has one compositional `VocoderTask`, so it does not delete fields according to a task scope. A resolved recipe is validated as one complete object instead. Reflective Generator and Discriminator parameters remain deliberately open for experiments, while unknown fields anywhere else are rejected. Constructor arguments are never silently filtered.

Built-in names may contain a namespace, for example `historical/ft-hifigan`. Recipe discovery and resolution are recursive, while internal files whose names begin with `_` remain hidden. See [Migrated configurations](migrated-configs.md) for the mapping from pre-refactor files.

## Reflective model parameters

Generator and Discriminator use a common specification:

```yaml
model:
  generator:
    target: nsf_hifigan
    params:
      mini_nsf: false
      upsample_rates: [8, 8, 2, 2, 2]
  discriminators:
    mpd:
      target: mpd
      params:
        periods: [3, 5, 7, 11, 17, 23, 37]
```

`params` is deliberately open and may contain nested experimental backbone settings. Values are passed unchanged as `target(**params)`. The factory never silently filters unsupported keys.

Optimizer, scheduler, objective, optimization plan, augmentation, data, runtime, and checkpoint policy are typed configurations. They cannot import arbitrary Python targets.

## Dataset configuration

The standard 44.1 kHz singing setup is:

```yaml
data:
  processed_dir: data/junninghua-44k-128
  features:
    sample_rate: 44100
    num_mels: 128
    hop_size: 512
    n_fft: 2048
    win_size: 2048
    fmin: 40
    fmax: 16000
    f0_min: 65
    f0_max: 1100
    pitch_extractor: parselmouth
  validation_size: 10
  seed: 114514
```

`pitch_extractor` accepts `parselmouth` and `harvest`. Unsupported values, including unimplemented RMVPE extraction, are rejected by validation.

## Optimization plans

Each optimization phase declares one or more parameter routes. AdamW can own all parameters, or a
Muon route can be paired with an AdamW route for the remaining parameters. Every optimizer is an
independent PyTorch optimizer with its own state and optional scheduler:

```yaml
optimization:
  name: alternating
  generator:
    optimizers:
      - parameters: all
        optimizer: {name: adamw, lr: 0.0001}
        scheduler: null
    frequency: 1
    accumulate: 1
    clip_grad_norm: 1
  discriminator:
    optimizers:
      - parameters: muon
        optimizer: {name: muon, lr: 0.0002}
      - parameters: remaining
        optimizer: {name: adamw, lr: 0.0002}
    frequency: 1
    accumulate: 1
    clip_grad_norm: 1
```

The built-in Muon route preserves the original project selection rule: trainable tensors with at
least two dimensions use Muon, except embeddings and parameters named `weight_g`; all other tensors
use AdamW. The engine verifies that routes are disjoint and cover every trainable parameter. It runs
one backward pass for the phase, steps both optimizers at the same accumulation boundary, and saves
their states separately.

Schedulers default to disabled. Historical WarmupLR fields that were never registered are not inferred or enabled.
