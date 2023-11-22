# SingingVocoders
A collection of neural vocoders suitable for singing voice synthesis tasks.

# Quick Start

## processing
Frist ,you need to use [process.py](process.py) to preprocess your data

The following configuration items are what you need to change during preprocessing

```angular2html

data_input_path: []  the path for your data

data_out_path: [] the path for preprocess out put

val_num: 1 the number of validation audio
```
examples
```
data_input_path: ['wav/in1','wav/in2'] 

data_out_path: ['wav/out1','wav/out2']
val_num: 5 # This is the number of valves you want. 

 # (The files are automatically extracted during preprocessing.)

 # (The paths in the two lists are one-to-one, so the number should be the same.)


 # (Then, the preprocessor scans all .wav files, including subfolders.)

 # (Normally, there are only these three to change.)
```

and running preprocessing scripts
```angular2html
[process.py](process.py) --config (your config path) --num_cpu (Number of cpu threads used in preprocessing)  --strx (1 for a forced absolute path 0 for a relative path)

```

## training
running training scripts
```angular2html
[train.py](train.py) --config (your config path) --exp_name (your ckpt name) --work_dir Working catalogue (optional)

```
## export checkpoint
if you finish training you can use this scripts to export diffsinger vocode checkpoint
```
[export_ckpt.py](export_ckpt.py)export_ckpt.py --exp_name (your ckpt name)  --save_path (output ckpt path) --work_dir Working catalogue (optional)
```



# 注意

因为pl的问题所以说再gan里面实际的步数是 他显示的 //2

如果你需要 微调社区vocode建议使用[ft_hifigan.yaml](configs%2Fft_hifigan.yaml) 配置文件

如何使用 微调功能 建议参考 ds文档

少量步数的微调可以 冻结mpd

建议不要用 bf16 可能会产生音质问题

差不多2k step就可以微调完成





































