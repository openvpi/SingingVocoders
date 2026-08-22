# SingingVocoders

SingingVocoders 是面向歌声音频的神经声码器训练框架。训练运行时基于 Lightning Fabric，原生支持 GAN 的多模型、多优化器、交替更新、预热、one-step joint、梯度累积、AMP 与 DDP；项目不依赖 PyTorch Lightning Trainer。

[English documentation](README_en.md)

## 安装

Windows 环境默认从 Conda explicit lock 和带 SHA-256 的 Python lock 创建。脚本发现同名环境时会直接停止，不会修改或删除已有环境；仅在 Conda 已成功创建新环境、而后续 Python 安装失败时清理该新环境。若 Conda 创建本身失败，脚本不会按名称删除任何环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\create_env.ps1
conda activate singing-vocoders-refactor
```

环境固定为 Python 3.11、PyTorch/Torchaudio 2.11.0 CUDA 13.0、Lightning Fabric 2.6.5 和 NumPy 1.26.4。`conda-win-64.lock` 固定 Conda URL 与 SHA-256，`pylock.win-64.toml` 固定每个 Python 构件及 SHA-256；`environment.yml` 仅作为需要在线求解时的开发规格，不是复现安装入口。安装后只会注册一个命令：`vocoder`。

## 快速开始

查看并验证内置 recipe：

```powershell
vocoder recipes list
vocoder config validate pc-nsf-hifigan
```

预处理 WAV 数据：

```powershell
vocoder preprocess pc-nsf-hifigan `
  --source D:\path\to\wavs `
  --output data\my-dataset
```

预处理通过同级 staging 目录事务发布。已有输出只有在完整通过当前数据格式校验时才能被替换；普通目录、残缺数据集、符号链接和 junction 都会被拒绝。

训练配置可以用重复的 `--set key=value` 覆盖。每次训练都创建全新的运行目录，不覆盖已有结果：

```powershell
vocoder train pc-nsf-hifigan `
  --set data.processed_dir=data/my-dataset `
  --set runtime.precision=bf16-mixed
```

精确恢复新格式 checkpoint：

```powershell
vocoder train pc-nsf-hifigan --resume experiments\run\step-00000500.ckpt
```

严格导入旧 Lightning 或生成器权重时使用 warm-start；旧 optimizer、scaler 和 step 不会被猜测或恢复：

```powershell
vocoder train pc-nsf-hifigan `
  --warm-start pretrained\model.ckpt `
  --warm-start-scope all
```

验证、审计和导出：

```powershell
vocoder validate pc-nsf-hifigan experiments\run\step-00001000.ckpt
vocoder checkpoint inspect experiments\run\step-00001000.ckpt
vocoder export experiments\run\step-00001000.ckpt exports\model.pt
```

新格式 checkpoint 始终使用其内嵌配置导出。旧 checkpoint 需要额外传入 `--recipe pc-nsf-hifigan`。导出结果为 DiffSinger 使用的 `{"generator": state_dict}` checkpoint，以及同目录的 `config.json`。

## 内置 recipe

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
- `pc-nsf-hifigan`：`nsf-hifigan` 的 PC-NSF 配置叠层

配置按 `bases` 从左到右合并，再应用当前文件，最后应用 CLI dotlist。Generator 和 Discriminator 使用 `target + params` 反射配置；其他模块均使用强类型配置和显式注册表。未知字段、循环继承、未解析插值、缺失参数和多余参数都会报错。

一次完整 recipe iteration 记为一个 `logical_step`。Checkpoint 另外记录每个 optimizer 的 attempted/successful updates，因此 step 不再随 G/D optimizer 数量翻倍。

更多设计和约束见：

- [架构](docs/architecture.md)
- [配置与 recipe](docs/configuration.md)
- [checkpoint 与恢复](docs/checkpoints.md)

## 开发检查

```powershell
ruff check .
ruff format --check .
mypy
pytest
pip check
```

DDP 由 Fabric 引擎支持，并在 Windows 上以两进程 CPU/Gloo 回归测试覆盖同步、累积、计数器、单 rank checkpoint 写入与恢复。多卡 GPU 验收应在独立环境执行。
