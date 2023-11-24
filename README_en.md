# SingingVocoders
A collection of neural vocoders suitable for singing voice synthesis tasks.

# Quick Start

## Preprocessing


The following configuration items are what you need to change during preprocessing

```angular2html

data_input_path: []  the path for your data

data_out_path: [] the path for preprocess out put

val_num: 1 the number of validation audio
```
An example
```
data_input_path: ['wav/in1','wav/in2'] 

data_out_path: ['wav/out1','wav/out2']
val_num: 5 # This is the number of valves you want. 

 # (The files are automatically extracted during preprocessing.)

 # (The paths in the two lists are one-to-one, so the number should be the same.)


 # (Then, the preprocessor scans all .wav files, including subfolders.)

 # (Normally, there are only these three to change.)
```
It is recommended to modify it in [base.yaml](configs%2Fbase.yaml),
then run the following preprocessing script
```angular2html
python process.py --config (your config path) --num_cpu (Number of cpu threads used in preprocessing)  --strx (1 for a forced absolute path 0 for a relative path)

```

## Training
Run the training following script
```angular2html
python train.py --config (your config path) --exp_name (your ckpt name) --work_dir Working catalogue (optional)

```
## Export the checkpoint
if you finish training you can use this script to export the diffsinger vocoder checkpoint
```
python export_ckpt.py --exp_name (your ckpt name)  --save_path (output ckpt path) --work_dir Working catalogue (optional)
```

# Offline data augmentation
Replace the preprocessing script with [process_aug.py](process_aug.py) and add configuration entries
```
key_aug: false (Do not augment during training)
aug_min: 0.9  (Minimum f0 adjustment multiplier)
aug_max: 1.4   (Maximum f0 adjustment multiplier)
aug_num: 1   (Data augmentation multiplier)
```
That's it. Note that data augmentation may damage the sound quality!
# Online data augmentation (recommend)
Note that to use the online data augmentation, use the [process.py](process.py) script, otherwise offline and online augmentation will be superimposed
```angular2html
key_aug: true (Do augment during training)
key_aug_prob: 0.5 (Data augmentation probability)
aug_min: 0.9  (Minimum f0 adjustment multiplier)
aug_max: 1.4   (Maximum f0 adjustment multiplier)
```
Note that data augmentation may damage the sound quality!
# Note

Because of some problems the actual number of steps is half of what he shows

If you need to fine-tune the vocoder suggests using the [ft_hifigan.yaml](configs%2Fft_hifigan.yaml) 

How to use the fine-tuning function is recommended to refer to the openvpi/diffsinger project documentation

A small number of steps of fine-tuning can freeze the mpd module.

It is not recommended to use bf16, as it may cause sound quality problems.

Almost 2k steps is enough for fine-tuning of small dataset.





































