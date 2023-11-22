# SingingVocoders
A collection of neural vocoders suitable for singing voice synthesis tasks.


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





































