# Architecture

SingingVocoders separates model behavior from execution mechanics. The dependency direction is intentionally one-way:

```text
CLI / runtime orchestration
        |
        +-- configuration -> component factory
        +-- data pipeline -> Batch
        +-- VocoderTask -> generator + discriminators + objective
        +-- FabricGanEngine -> optimization plan + runtime state
        +-- checkpoint / export
```

## Component boundary

The public tensor contracts are:

- `Batch(audio, mel, f0, uv, pe, metadata)`
- `GeneratorOutput(audio, aux)`
- `DiscriminatorOutput(scores, feature_maps)`

Only Generator and Discriminator are reflective components. A `ComponentSpec` contains a `target` and an open `params` mapping. Built-in short names resolve through `ComponentRegistry`; trusted configurations may also use a dotted Python path. Before construction, the factory checks the callable signature and rejects missing or unexpected arguments unless the target explicitly accepts `**kwargs`. The constructed object must implement the requested component contract.

PC-NSF HiFiGAN and its HiFiGAN discriminator ensemble preserve their historical registered names, including `weight_g`/`weight_v` weight-normalization keys and `weight_orig`/`weight_u`/`weight_v` spectral-normalization keys. Other backbones use cleaned module layouts and do not promise legacy state-key compatibility.

## Task and objective

`VocoderTask` is the only training Task. It owns a configured generator, a discriminator ensemble, and one typed `VocoderObjective`. Recipe differences are composition, not subclasses.

Objectives expose auxiliary, discriminator, generator, and joint losses. They do not step optimizers, schedule learning rates, perform distributed synchronization, or save state. PC augmentation is a generator-stage transform over the common `Batch`; UV-dependent objectives consume the same contract.

## Fabric GAN engine

`FabricGanEngine` owns device placement, precision, distributed wrapping, backward synchronization, clipping, optimizer/scheduler stepping, and counters. Three typed optimization plans are available:

1. `alternating`: detached fake audio for D, then frozen D parameters with connected fake audio for G.
2. `warmup_alternating`: generator auxiliary updates until `aux_steps`, then the alternating protocol.
3. `joint`: one optimizer updates the combined G/D graph for the one-step objective.

Each optimizer slot independently controls frequency, accumulation, clipping, optimizer, and scheduler. During accumulation, `Fabric.no_backward_sync` suppresses premature DDP synchronization. A checkpoint may only be written when every optimizer is at an accumulation boundary.

One completed recipe iteration increments `logical_step` once. Each optimizer separately records micro steps, attempted updates, and successful updates. AMP overflow therefore does not corrupt the logical clock or masquerade as a successful optimizer update.

The D phase always detaches generated audio. The G phase freezes discriminator parameters while retaining the derivative through discriminator input to generated audio. Non-finite losses and gradients raise an error containing the phase and available batch identity.

## Data pipeline

Preprocessing recursively discovers WAV files, decodes them to float, averages channels, resamples once, and extracts mel, F0, and UV from the same target-rate waveform. Each NPZ has exactly `audio`, `mel`, `f0`, `uv`, and `pe`.

`dataset.json`, `manifest.jsonl`, `train.txt`, and `valid.txt` form the versioned dataset contract. Validation selection is based on the seeded hash of normalized relative paths. The fingerprint covers source and NPZ hashes, splits, audio metadata, feature settings and implementation version, and frame counts. A complete staging dataset is validated before publication. An existing output is replaced only when it is itself a fully valid dataset of the current format; unrelated, incomplete, linked, and reparse-point directories are rejected.

Training order and crops are deterministic per seed and epoch. Invalid samples raise immediately; the loader never recursively substitutes another sample.

## State boundary

Versioned checkpoints contain the complete Task, every optimizer and scheduler, Fabric precision state, logical/update counters, per-rank RNG, sampler state and position, resolved configuration, and data fingerprint. Exact resume preflights configuration, data, world size, sampler position, optimizer slots, and model tensor schema before loading live state.

Legacy checkpoint support is intentionally one-way: it strictly imports named, shape-compatible weights and starts a new optimizer/runtime state. See [checkpoints.md](checkpoints.md).
