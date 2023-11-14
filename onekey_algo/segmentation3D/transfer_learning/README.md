# Onekey Med3D迁移学习

## 参数配置

```shell
$ python train.py -h
usage: train.py [-h] [--data_root DATA_ROOT] [--img_list IMG_LIST]
                [--n_seg_classes N_SEG_CLASSES]
                [--learning_rate LEARNING_RATE] [--num_workers NUM_WORKERS]
                [--batch_size BATCH_SIZE] [--phase PHASE]
                [--save_intervals SAVE_INTERVALS] [--n_epochs N_EPOCHS]
                [--input_D INPUT_D] [--input_H INPUT_H] [--input_W INPUT_W]
                [--resume_path RESUME_PATH] [--pretrain_path PRETRAIN_PATH]
                [--new_layer_names NEW_LAYER_NAMES] [--no_cuda]
                [--gpu_id GPU_ID [GPU_ID ...]] [--model MODEL]
                [--model_depth MODEL_DEPTH]
                [--resnet_shortcut RESNET_SHORTCUT]
                [--manual_seed MANUAL_SEED] [--ci_test]

optional arguments:
  -h, --help            show this help message and exit
  --data_root DATA_ROOT
                        Root directory path of data
  --img_list IMG_LIST   Path for image list file
  --n_seg_classes N_SEG_CLASSES
                        Number of segmentation classes
  --learning_rate LEARNING_RATE
                        Initial learning rate (divided by 10 while training by
                        lr scheduler)
  --num_workers NUM_WORKERS
                        Number of jobs
  --batch_size BATCH_SIZE
                        Batch Size
  --phase PHASE         Phase of train or test
  --save_intervals SAVE_INTERVALS
                        Interation for saving model
  --n_epochs N_EPOCHS   Number of total epochs to run
  --input_D INPUT_D     Input size of depth
  --input_H INPUT_H     Input size of height
  --input_W INPUT_W     Input size of width
  --resume_path RESUME_PATH
                        Path for resume model.
  --pretrain_path PRETRAIN_PATH
                        Path for pretrained model.
  --new_layer_names NEW_LAYER_NAMES
                        New layer except for backbone
  --no_cuda             If true, cuda is not used.
  --gpu_id GPU_ID [GPU_ID ...]
                        Gpu id lists
  --model MODEL         (resnet | preresnet | wideresnet | resnext | densenet
                        |
  --model_depth MODEL_DEPTH
                        Depth of resnet (10 | 18 | 34 | 50 | 101)
  --resnet_shortcut RESNET_SHORTCUT
                        Shortcut type of resnet (A | B)
  --manual_seed MANUAL_SEED
                        Manually set random seed
  --ci_test             If true, ci testing is used.
```

#### 参数的含义

* data_root，数据存放的路径。
* img_list，每一条训练数据一行。每一个样本为img_path，与data_root拼接起来形成一个完整样本。data_root/img_path
* n_seg_classes，待分割的数据的类别数。这个地方需要根据自己数据集情况修改。
* learning_rate，学习率。
* num_workers，加载数据集的线程数，越大越快，但是注意不要超过机器CPU数。
* batch_size，batch size。
* phase，阶段。train | valid
* save_intervals，训练多少次保存一次模型。
* n_epoch，训练的epoch数。
* input_D、input_H，input_W，输入数据的通道数、高度、宽度。
* resume_path，断点重新训练。
* pretrain_path，与训练的模型。在pretrain文件夹中的模型都是可选的。
* gpu_id，使用的GPU ID。
* model，使用的模型，默认是Resnet。
* model_depth，模型的深度，可选，10，18，34，50，101。
* resnet_shortcut，shortcut类型，配合model_depth一起使用。
* ci_test，是否进行测试。

### 修改参数

这些参数可以使用命令行进行指定，例如

```shell
python train.py --data_root path2Udata --num_workers 2
```

也可以通过修改某一个参数的default值来修改。例如

```python
parser.add_argument('--pretrain_path', default='pretrain/resnet_18.pth', type=str,
                    help='Path for pretrained model.')
```

将预训练的模型修改成`resnet_50_23dataset.pth`。

```python
parser.add_argument('--pretrain_path', default='pretrain/resnet_50_23dataset.pth', type=str,
                    help='Path for pretrained model.')
```

最终模型保存在trails/models/目录底下。

### 训练模型

1. 按需修改setting.py中间的参数。

2. 运行train.py或者使用命令行

   ```shell
   python train.py --gpu_id 0 1    # multi-gpu training on gpu 0,1
   # or
   python train.py --gpu_id 0    # single-gpu training on gpu 0
   ```

### 测试模型

1. 加载训练好的模型。通过修改`resume_path`参数指定

   ```python
   parser.add_argument('--resume_path', default='path2Umodel', type=str, help='Path for resume model.')
   ```

2. 指定测试数据。通过 `image_list`参数指定。

   ```python
   parser.add_argument('--img_list', default='./data/train.txt', type=str, help='Path for image list file')
   ```

例如这两个参数修改为

```python
parser.add_argument('--resume_path', default='trails/models/resnet_50_epoch_40_batch_0.pth.tar', 
                    type=str, help='Path for resume model.')
parser.add_argument('--img_list', default='./data/val.txt', type=str, help='Path for image list file')
```

## 常见错误

**错误一**：`RuntimeError: CUDA out of memory. `

此时可以调小四个参数，可以逐渐减小一半

* batchsize，一次训练的批次。
* input_D，输入的通道数。
* input_H，输入的图像的高度。
* input_W，输入的图像的宽度。

**错误二**：模型正确运行，但是loss开始比较大。

```shell
INFO	Batch: 0-0 (0), loss = 1.651, loss_seg = 1.651, avg_batch_time = 2.007
INFO	Batch: 0-1 (0), loss = 2.189, loss_seg = 2.189, avg_batch_time = 3.979
INFO	Batch: 0-2 (0), loss = 0.780, loss_seg = 0.780, avg_batch_time = 5.555
```

这时注意模型的配置预shortcut类型的一致程度。

## 更新(2021/07/01)

使用了更多的数据集进行与训练(23 datasets)，下面是参数的配置。
```
Model name             : parameters settings
resnet_10_23dataset.pth: --model resnet --model_depth 10 --resnet_shortcut B
resnet_18_23dataset.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet_34_23dataset.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet_50_23dataset.pth: --model resnet --model_depth 50 --resnet_shortcut B

# 下面是更新之前的模型深度和shortcut类型的配置。
resnet_10.pth: --model resnet --model_depth 10 --resnet_shortcut B
resnet_18.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet_34.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet_50.pth: --model resnet --model_depth 50 --resnet_shortcut B
resnet_101.pth: --model resnet --model_depth 101 --resnet_shortcut B
resnet_152.pth: --model resnet --model_depth 152 --resnet_shortcut B
resnet_200.pth: --model resnet --model_depth 200 --resnet_shortcut B

```

上面的模型，我们使用了更多的数据集进行与训练，下面是模型精度的对比。

| **Network** | **Pretrain**       | **LungSeg(Dice)** |
| ----------- | ------------------ | ----------------- |
| 3D-ResNet10 | Train from scratch | 69.31%            |
| resnet_10_23dataset | MedicalNet  | 96.56%                          |
| 3D-ResNet18 | Train from scratch | 70.89%            |
| resnet_18_23dataset | MedicalNet  | 94.68%                           |
| 3D-ResNet34 | Train from scratch | 75.25%            |
| resnet_34_23dataset | MedicalNet  | 94.14%                           |
| 3D-ResNet50 | Train from scratch | 52.94%            |
| resnet_50_23dataset | MedicalNet  | 89.25%                            |

原始部分数据集联合训练的模型

| Network      | Pretrain           | LungSeg(Dice) | NoduleCls(accuracy) |
| ------------ | ------------------ | ------------- | ------------------- |
| 3D-ResNet10  | Train from scratch | 71.30%        | 79.80%              |
| | MedicalNet   | 87.16%             | 86.87%                           |
| 3D-ResNet18  | Train from scratch | 75.22%        | 80.80%              |
|     | MedicalNet   | 87.26%             | 88.89%                      |
| 3D-ResNet34  | Train from scratch | 76.82%        | 83.84%              |
|  | MedicalNet   | 89.31%             | 89.90%                         |
| 3D-ResNet50  | Train from scratch | 71.75%        | 84.85%              |
|    | MedicalNet   | 93.31%             | 89.90%                       |
| 3D-ResNet101 | Train from scratch | 72.10%        | 81.82%              |
| | MedicalNet   | 92.79%             | 90.91%                           |
| 3D-ResNet152 | Train from scratch | 73.29%        | 73.74%              |
|   | MedicalNet   | 92.33%             | 90.91%                        |
| 3D-ResNet200 | Train from scratch | 71.29%        | 76.77%              |
|  | MedicalNet   | 92.06%             | 90.91%                          |

## 验证与训练模型是否生效

预训练模型生效与否，主要看模型开始阶段训练的loss情况，以及训练过程中loss下降情况。

* 预训练模型生效时loss开始更小，下降更快。

### train from scratch（从头开始训练）

```shell
INFO	Batch: 0-1 (0), loss = 1.722, loss_seg = 1.722, avg_batch_time = 4.220
INFO	Batch: 0-2 (0), loss = 1.569, loss_seg = 1.569, avg_batch_time = 5.959
INFO	Batch: 0-3 (0), loss = 1.455, loss_seg = 1.455, avg_batch_time = 7.674
INFO	Batch: 0-4 (0), loss = 1.425, loss_seg = 1.425, avg_batch_time = 9.399
INFO	Batch: 0-5 (0), loss = 1.472, loss_seg = 1.472, avg_batch_time = 11.124
INFO	Batch: 0-6 (0), loss = 1.340, loss_seg = 1.340, avg_batch_time = 12.861
INFO	Batch: 0-7 (0), loss = 1.212, loss_seg = 1.212, avg_batch_time = 14.589
INFO	Batch: 0-8 (0), loss = 1.366, loss_seg = 1.366, avg_batch_time = 16.318
INFO	Batch: 0-9 (0), loss = 1.004, loss_seg = 1.004, avg_batch_time = 18.115
```

### 预训练模型生效

```shell
INFO	Batch: 0-0 (0), loss = 0.934, loss_seg = 0.934, avg_batch_time = 1.995
INFO	Batch: 0-1 (0), loss = 0.553, loss_seg = 0.553, avg_batch_time = 4.202
INFO	Batch: 0-2 (0), loss = 0.456, loss_seg = 0.456, avg_batch_time = 5.920
INFO	Batch: 0-3 (0), loss = 0.560, loss_seg = 0.560, avg_batch_time = 7.626
INFO	Batch: 0-4 (0), loss = 0.499, loss_seg = 0.499, avg_batch_time = 9.338
INFO	Batch: 0-5 (0), loss = 0.473, loss_seg = 0.473, avg_batch_time = 11.054
INFO	Batch: 0-6 (0), loss = 0.395, loss_seg = 0.395, avg_batch_time = 12.776
INFO	Batch: 0-7 (0), loss = 0.382, loss_seg = 0.382, avg_batch_time = 14.499
INFO	Batch: 0-8 (0), loss = 0.369, loss_seg = 0.369, avg_batch_time = 16.229
INFO	Batch: 0-9 (0), loss = 0.336, loss_seg = 0.336, avg_batch_time = 17.967
```

