# SingingVocoders
A collection of neural vocoders suitable for singing voice synthesis tasks.

# English version [README_en.md](README_en.md)
## If you have any questions, please open an issue.
# 训练

[train.py](train.py) --config 配置文件 --exp_name ckpt名字 --work_dir 工作目录（可选）

# 预处理 

[process.py](process.py) --config 配置文件 --num_cpu 并行数量 --strx 1代表 强制绝对路径 0代表相对路径


和预处理有关的配置文件
```
DataIndexPath: dataX11   这个是训练数据index的位置 预处理会自动生成

valid_set_name: validX 这个是val index的名字 预处理会自动生成

train_set_name: trainX 这个是训练的 index的名字 预处理会自动生成

#data_input_path: []  这个是你的wav的输入目录

#data_out_path: []这个是你的npz的输出目录  预处理之后的格式是npz



val_num: 1 这个是你要的 val 数量 
```

例子
```
#data_input_path: [’wav/in1‘,’wav/in2‘] 这个是你的wav的输入目录

#data_out_path: [’wav/out1‘,’wav/out2‘]这个是你的npz的输出目录
val_num: 5 这个是你要的 val 数量 预处理的时候会自动抽取 文件
两个列表里面的 路径是一一对应的所以说 数量要一样

然后 预处理的时候会扫描全部的.wav文件 包括子文件夹的

正常情况下只有这三个要改
```
# 导出
[export_ckpt.py](export_ckpt.py)export_ckpt.py --exp_name ckpt名字  --save_path 导出的ckpt --work_dir 工作目录（可选） 

# 注意

因为pl的问题所以说再gan里面实际的步数是 他显示的 //2

如果你需要 微调社区vocode建议使用[ft_hifigan.yaml](configs%2Fft_hifigan.yaml) 配置文件

如何使用 微调功能 建议参考 ds文档

少量步数的微调可以 冻结mpd

建议不要用 bf16 可能会产生音质问题

差不多2k step就可以微调完成

# 快速开始
## 预处理
以下是你需要根据自己的数据集修改的配置项
```angular2html

data_input_path: []  这个列表 是你原始wav文件的路径

data_out_path: [] 此列表 预处理输出的npz文件的路径

val_num: 1 这个是在验证的时候 抽取的音频文件数量
```
运行预处理
```angular2html
process.py --config (your config path) --num_cpu (Number of cpu threads used in preprocessing)  --strx (1 for a forced absolute path 0 for a relative path)

```
## 训练
```angular2html
train.py --config (your config path) --exp_name (your ckpt name) --work_dir Working catalogue (optional)

```
## 导出
```angular2html
export_ckpt.py --exp_name (your ckpt name)  --save_path (output ckpt path) --work_dir Working catalogue (optional)
```
# 注意事项
实际步数是显示的//2

微调请使用[ft_hifigan.yaml](configs%2Fft_hifigan.yaml)

微调功能使用 请参考diffsinger 文档

不要使用bf16训练模型 他可能导致音质问题

2k step左右即可微调完成

冻结mpd可能可以有更好的结果





































