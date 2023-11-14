## run_segmentation

```shell
$ python run_segmentation.py -h
usage: run_segmentation.py [-h] [--dataset DATASET] --data_path DATA_PATH
                           [--model MODEL] [--aux-loss] [--device DEVICE]
                           [-b BATCH_SIZE] [--epochs N] [-j N] [--lr LR]
                           [--optimizer OPTIMIZER] [--momentum M] [--wd W]
                           [--print-freq PRINT_FREQ] [--output-dir save_dir]
                           [--resume RESUME] [--test-only] [--pretrained]
                           [--world-size WORLD_SIZE] [--dist-url DIST_URL]

PyTorch Segmentation Training

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset
  --data_path DATA_PATH
                        Root data path.
  --model MODEL         model
  --aux-loss            auxiliary loss
  --device DEVICE       device
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  --epochs N            number of total epochs to run
  -j N, --workers N     number of data loading workers (default: 16)
  --lr LR               initial learning rate
  --optimizer OPTIMIZER
                        Optimizer
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --print-freq PRINT_FREQ
                        print frequency
  --output-dir save_dir
                        path where to save
  --resume RESUME       resume from checkpoint
  --test-only           Only test the model
  --pretrained          Use pre-trained models from the modelzoo
  --world-size WORLD_SIZE
                        number of distributed processes
  --dist-url DIST_URL   url used to set up distributed training_a_classifier
```

* `dataset`：如果是coco格式的数据集，一定要以 `_coco_fmt`结尾。
* `data_path`：原始数据存放路径，包括`train`，`val`文件夹。
* `print-freq`：多少次迭代打印一次log。

### labelme标注

> 安装：[https://github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)
> 使用文档：[https://blog.csdn.net/s534435877/article/details/104353804](https://blog.csdn.net/s534435877/article/details/104353804)

### convert2coco数据格式
```shell script
cd ../scripts
python convet2coco.py --input_dir $LABEL_ROOT --save_dir $path2save --labels labels.txt
```
* `--labels`：对应的标签列表，需要自行创建，来源于labelme中的标签集合。

### 训练
```shell script
python run_segmentation.py --dataset skin_coco_fmt --data_path $path2save -j 0
```

### 预测
```shell script
python eval_segmentation.py --data test_case/* --resume model_2.pth --class_name labels.txt
```

### 多GPU卡训练
1. fcn_resnet101
    ```
    python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --lr 0.02 --dataset coco -b 4 --model fcn_resnet101 --aux-loss
    ```

2. deeplabv3_resnet101
    ```
    python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --lr 0.02 --dataset coco -b 4 --model deeplabv3_resnet101 --aux-loss
    ```
