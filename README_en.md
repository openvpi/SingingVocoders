# SingingVocoders
A collection of neural vocoders suitable for singing voice synthesis tasks.

# Quick Start

## processing


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
process.py --config (your config path) --num_cpu (Number of cpu threads used in preprocessing)  --strx (1 for a forced absolute path 0 for a relative path)

```

## training
running training scripts
```angular2html
train.py --config (your config path) --exp_name (your ckpt name) --work_dir Working catalogue (optional)

```
## export checkpoint
if you finish training you can use this scripts to export diffsinger vocode checkpoint
```
export_ckpt.py --exp_name (your ckpt name)  --save_path (output ckpt path) --work_dir Working catalogue (optional)
```

# Data enhancement
Replace the preprocessing script with [process_aug.py](process_aug.py) and add configuration entries
```
aug_min: 0.9  Minimum Transpose Multiplier
aug_max: 1.4   Maximum Variable Adjustment Multiplier
aug_num: 5   Data Enhancement Multiplier
```
That's it.

# note

Because of some problems the actual number of steps is what he shows //2

If you need to fine-tune the community vocode suggests using the[ft_hifigan.yaml](configs%2Fft_hifigan.yaml) 

How to use the fine-tuning function Suggested reference ds documentation

A small number of steps of fine tuning can freeze the mpd.

It is not recommended to use bf16, as it may cause sound quality problems.

Almost 2k steps is enough for fine tuning.





































