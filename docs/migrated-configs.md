# Migrated configurations

The configurations tracked before the Fabric refactor remain available as typed recipe overlays. Their public names are namespaced under `historical/` so they cannot be confused with the maintained canonical recipes.

| Previous path | Converted recipe | Canonical task |
| --- | --- | --- |
| `configs/base.yaml` | `historical/_base.yaml` | Shared preprocessing defaults; not directly runnable |
| `configs/base_hifi.yaml` | `historical/base-hifi` | `nsf-hifigan` |
| `configs/base_hifi_one.yaml` | `historical/base-hifi-one` | `one-step-nsf-hifigan` |
| `configs/ft_hifigan.yaml` | `historical/ft-hifigan` | `pc-nsf-hifigan` |
| `configs/nsf_hifigan_fast.yaml` | `historical/nsf-hifigan-fast` | `fast-nsf-hifigan` |
| `configs/base_ddspgan.yaml` | `historical/base-ddspgan` | `ddspgan` |
| `configs/lvc_base_ddspgan.yaml` | `historical/lvc-base-ddspgan` | `lvc-ddspgan` |
| `configs/ddsp_univnet.yaml` | `historical/ddsp-univnet` | `ddsp-univnet` |
| `configs/nsf_univnet.yaml` | `historical/nsf-univnet` | `nsf-univnet` |
| `configs/univnet.yaml` | `historical/univnet` | `univnet` |
| `configs/HiFivae.yaml` | `historical/hifivae` | `hifivae` |

The overlays preserve effective model dimensions, discriminator periods, optimizer values, crop and batch sizes, validation sizes, runtime limits, and objective weights where the new framework has the same concept. They inherit maintained component definitions rather than copying complete model configurations.

Several old fields intentionally become workflow operations or explicit migrations:

- `task_cls` is represented by the generator target, objective, and optimization plan.
- `finetune_ckpt_path` is not executable configuration. Reproduce the previous fine-tune preset with `vocoder train historical/ft-hifigan --warm-start pretrained/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt --warm-start-scope all`.
- The old RMVPE value is migrated to Parselmouth because RMVPE is not implemented by the current deterministic preprocessing pipeline. Harvest remains selectable with `--set data.features.pitch_extractor=harvest`.
- The old preprocessing-time `key_aug` fields have no equivalent in the current online augmentation contract and are not silently mapped to PC augmentation.
- The historical WarmupLR entries were never attached to a Lightning optimizer and remain disabled instead of changing training behavior.

Every converted recipe passes the same strict OmegaConf/Pydantic validation and component-construction checks as the canonical recipes.
