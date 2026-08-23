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

Alternating slots may use `adamw` or `muon_adamw` and optionally an exponential scheduler:

```yaml
optimization:
  name: alternating
  generator:
    optimizer: {name: adamw, lr: 0.0001}
    frequency: 1
    accumulate: 1
    clip_grad_norm: 1
    scheduler: null
  discriminator:
    optimizer: {name: adamw, lr: 0.0001}
    frequency: 1
    accumulate: 1
    clip_grad_norm: 1
    scheduler: null
```

Muon+AdamW sends trainable two-dimensional matrix parameters through Muon and all remaining parameters through AdamW. The partition is stable and included in optimizer checkpoint state.

Schedulers default to disabled. Historical WarmupLR fields that were never registered are not inferred or enabled.
