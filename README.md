# SingingVocoders
A collection of neural vocoders suitable for singing voice synthesis tasks.


# 训练

[train.py](train.py) --config 配置文件 --exp_name ckpt名字 --work_dir 工作目录（可选）

# 预处理 

[process.py](process.py) --config 配置文件 --num_cpu 并行数量 --strx 1代表 强制绝对路径 0代表相对路径




DataIndexPath: dataX11   这个是训练数据index的位置

valid_set_name: validX 这个是val index的名字

train_set_name: trainX 这个是训练的 index的名字

#data_input_path: []  这个是你的wav的输入目录

#data_out_path: []这个是你的wav的输出目录

val_num: 1 这个是你要的 val 数量 

# 导出
[export_ckpt.py](export_ckpt.py)export_ckpt.py --exp_name ckpt名字  --save_path 导出的ckpt --work_dir 工作目录（可选） 





































