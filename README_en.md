# SingingVocoders

SingingVocoders is a neural vocoder training framework for singing voice audio. Its runtime is built on Lightning Fabric and natively supports multi-model GAN training, multiple optimizers, alternating updates, warm-up phases, one-step joint optimization, gradient accumulation, AMP, and DDP. It does not depend on the PyTorch Lightning Trainer.

[中文文档](README.md)

## Installation

Create Windows environments from the Conda explicit lock and the SHA-256-verified Python lock. The script refuses to touch an existing environment. It removes the new environment only after Conda creation succeeded and the subsequent Python installation failed; a failed Conda creation never triggers deletion by environment name:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\create_env.ps1
conda activate singing-vocoders-refactor
```

The environment pins Python 3.11, PyTorch/Torchaudio 2.11.0 with CUDA 13.0, Lightning Fabric 2.6.5, and NumPy 1.26.4. `conda-win-64.lock` fixes Conda URLs and SHA-256 values; `pylock.win-64.toml` fixes every Python artifact and SHA-256. `environment.yml` is a solver-based development specification, not the reproducible installation entry point. The package installs exactly one command: `vocoder`.

## Quick start

List and validate recipes:

```powershell
vocoder recipes list
vocoder config validate pc-nsf-hifigan
```

Preprocess a recursive WAV dataset:

```powershell
vocoder preprocess pc-nsf-hifigan `
  --source D:\path\to\wavs `
  --output data\my-dataset
```

Preprocessing publishes through a sibling staging directory. An existing output can be replaced only if it fully validates as a dataset in the current format; ordinary directories, incomplete datasets, symbolic links, and junctions are rejected.

Repeated `--set key=value` options override resolved recipe values. Every training invocation creates a new run directory and refuses to overwrite an existing result:

```powershell
vocoder train pc-nsf-hifigan `
  --set data.processed_dir=data/my-dataset `
  --set runtime.precision=bf16-mixed
```

Resume an exact versioned checkpoint:

```powershell
vocoder train pc-nsf-hifigan --resume experiments\run\step-00000500.ckpt
```

Use warm-start for strict legacy Lightning or generator weights. Legacy optimizer, scaler, and step state are never inferred:

```powershell
vocoder train pc-nsf-hifigan `
  --warm-start pretrained\model.ckpt `
  --warm-start-scope all
```

Validate, inspect, and export:

```powershell
vocoder validate pc-nsf-hifigan experiments\run\step-00001000.ckpt
vocoder checkpoint inspect experiments\run\step-00001000.ckpt
vocoder export experiments\run\step-00001000.ckpt exports\model.pt
```

Versioned checkpoints always export with their embedded configuration. Legacy checkpoints additionally require `--recipe pc-nsf-hifigan`. DiffSinger export produces a `{"generator": state_dict}` checkpoint and a sibling `config.json`.

## Built-in recipes

- `nsf-hifigan`
- `fast-nsf-hifigan`
- `one-step-nsf-hifigan`
- `hifivae`
- `ddspgan`
- `lvc-ddspgan`
- `univnet`
- `ddsp-univnet`
- `nsf-univnet`
- `nsf-univnet-msd`
- `pc-nsf-hifigan`, a PC-NSF overlay on `nsf-hifigan`

Configuration merges `bases` from left to right, then the leaf file, and finally CLI dotlist overrides. Generator and Discriminator are reflective `target + params` components. All other subsystems use typed configurations and explicit registries. Unknown fields, inheritance cycles, unresolved interpolations, missing parameters, and unsupported extra parameters fail immediately.

One complete recipe iteration is one `logical_step`. Checkpoints separately record attempted and successful updates for every optimizer, so step counts no longer double for G/D updates.

Additional documentation:

- [Architecture](docs/architecture.md)
- [Configuration and recipes](docs/configuration.md)
- [Checkpointing and resume](docs/checkpoints.md)

## Development checks

```powershell
ruff check .
ruff format --check .
mypy
pytest
pip check
```

The Fabric engine supports DDP. A two-process CPU/Gloo test on Windows covers synchronization, accumulation, counters, single-rank checkpoint writing, and resume. Multi-GPU acceptance should run in a separate environment.
