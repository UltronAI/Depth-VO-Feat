# Depth-VO-Feat Pose Estimation Pytorch Version

### 1. nics_fix_pt

来源于妃神的定点训练的代码，这里拿来即用了，没做修改

### 2. example

妃神给的示例代码

### 3.  output

存储预测的位姿

### 4. from_caffe

从caffe模型转换过来的pytorch浮点模型

### 5. fixmodel.py

用来做定点finetune的模型，每一层都可通过参数决定是否要对该层定点

### 6. finetune.py

主要使用的脚本，在浮点模型的基础上进行定点finetune，或者接着定点模型继续finetune

比如可以使用如下命令：

`python finetune.py --checkpoint from_caffe/nyu_fix.pth.tar --epochs 30 -g0 --dataset-dir /home/share/kitti_odometry/dataset/ --output-dir checkpoints`

如果要修改定点模型，可以修改`input_fix`,`output_fix`,`conv_weight_fix`,`conv_output_fix`,`fc_weight_fix`,`fc_output_fix`，共6个conv层，3个fc层

### 7. inference.py

训好模型后，可以使用这个脚本获取的估计的位姿txt文件

`python inference.py --checkpoint checkpoints/$model -g0 --dataset-dir /home/share/kitti_odometry/dataset/ --output-dir output`

获取到文件之后就可以使用DSLAM_PySim做仿真了，绘图看下效果如何，具体操作看DSLAM_PySim中的说明

### 8. ../run.py

如果将pytorch模型转换成了caffe模型，可以使用run.py存储caffe模型输出的位姿，修改其中的caffe模型的路径，和存储路径即可