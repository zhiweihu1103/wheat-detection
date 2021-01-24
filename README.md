# wheat-detection
# Install Dependencies
```
conda create -n env_wheat
conda activate env_wheat
pip install torch==1.4.0
pip install cython
pip install torchvision==0.5.0
pip install albumentations
pip install imagecorruptions
pip install pycocotools
pip install terminaltables
pip install mmcv-full
sudo pip install -e .
```
The version of **mmdetection** we use is [2.7.0](https://codeload.github.com/open-mmlab/mmdetection/zip/v2.7.0), and the version of **mmcv-full** we use is [1.2.1](https://download.openmmlab.com/mmcv/dist/cu100/torch1.4.0/mmcv_full-1.2.1-cp38-cp38-manylinux1_x86_64.whl),The version of **python** we use is 3.8.5.

If you run **pip install mmcv-full** meet wrong notification, you can see [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions. In our case, we use following command to successfully install mmcv.
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu100/torch1.4.0/index.html
```
# Prepare in advance
* You should first process the data into voc2017 format and put it in the **data** path.
* For subsequent training, you should modify the file under **config**. In our case, we have modified the following parts:
  * For **configs\_base_\models\faster_rcnn_r50_fpn.py** file
    * change **num_classes=1** (line 46, 1 is represent the class nums);
  * For **configs\_base_\datasets\coco_detection.py** file
    * change **data_root = 'data/wheat/'** (line 2)
    * search **img_scale**, change it to (1024, 1024)
    * change **workers_per_gpu=0** (line 32)
    * change **samples_per_gpu=4** (line 31)
    * change all train/val/test related information, search **ann_file** and **img_prefix** to your datasets path.
  * For **\mmdet\datasets\coco.py** file
    * change **CLASSES = ('wheat')** (line 32)
  * For **\mmdet\core\evaluation\class_names.py** file
    * change **coco_classes** (line 69)
* Besides, for **faster_rcnn_r50_fpn_1x.py**, we also made the following changes to this config file:
```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=1)
total_epochs = 10
work_dir = './logs_wheat/faster_rcnn_r50/normal'
```
# Train
Take Faster R-CNN-R50 as example, you should cd the project root path, latter execute the following command
```
sh scripts/train_faster_rcnn_r50_fpn_1x.sh
```
You can see the logs by following command
```
tail -f logs_console/train_faster_rcnn_r50_fpn_1x.out
```
