# Checkpointing and resume

## Versioned checkpoints

Format version 1 stores:

- generator and discriminator weights;
- all optimizer and scheduler states;
- Fabric precision plugin and scaler state;
- `logical_step`, epoch, batch position, and per-optimizer counters;
- Python, NumPy, CPU Torch, and per-device CUDA RNG state for every rank;
- sampler identity and state;
- the resolved configuration and its fingerprints;
- the processed dataset fingerprint.

Writes use a temporary sibling file and an atomic replace. Only rank zero commits and prunes files, while all ranks participate in state collection and barriers. Saving is rejected unless every optimizer is at an accumulation boundary.

Exact resume permits changes only to `runtime.max_steps`, `runtime.log_every`, and `runtime.validate_every`. Model, data, augmentation, optimization, precision, checkpoint cadence, and dataset fingerprint must match. The engine resumes at the next unconsumed batch in the same epoch.

```powershell
vocoder train pc-nsf-hifigan `
  --resume experiments\pc-nsf-hifigan\run-id\step-00000500.ckpt `
  --set runtime.max_steps=1000
```

## Legacy weight import

The importer recognizes:

- Lightning mappings with `state_dict`;
- state-only task mappings;
- DiffSinger `{"generator": state_dict}` exports;
- raw generator state dictionaries.

All selected names and shapes are compared before `load_state_dict`. Import never mutates the source checkpoint and never restores legacy optimizer, scaler, or global-step state.

```powershell
vocoder train pc-nsf-hifigan `
  --warm-start pretrained\pc_nsf_hifigan.ckpt `
  --warm-start-scope generator
```

Use `--warm-start-scope all` only when the legacy checkpoint contains a generator and discriminator ensemble that exactly matches the recipe.

## Inspection and export

```powershell
vocoder checkpoint inspect path\to\checkpoint.ckpt
vocoder export path\to\checkpoint.ckpt exports\model.pt
vocoder export path\to\legacy.ckpt exports\legacy-model.pt --recipe pc-nsf-hifigan
```

Inspection uses metadata tensors and reports byte size, container kind, format version, key count, state tensor element count, dtypes, and a deterministic schema hash.

DiffSinger export is restricted to NSF HiFiGAN. A versioned checkpoint must use its embedded configuration; a legacy checkpoint requires an external recipe. Export writes `{"generator": state_dict}` and a sibling `config.json`. The JSON derives sample rate, mel count, hop, FFT/window sizes, frequency range, `pc_aug`, `mini_nsf`, and `noise_sigma` from the actual embedded data, augmentation, and generator settings.
