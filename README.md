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
sudo pip install -v -e .
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
# Test
Take Faster R-CNN-R50 as example, you should cd the project root path, latter execute the following command
```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/wheat/faster_rcnn_r50_fpn_1x.py logs_wheat/faster_rcnn_r50/normal/latest.pth --show-dir show_test/faster_rcnn_r50/normal --eval bbox
```
Then at **show_test/faster_rcnn_r50/normal** you will find the predict result with bbox.
# Results and Logs (AP)
Task | Backbone | Loss | schd | Att | DCN | fps | a-0.5 | a-0.75 | a-100-mul | s-mul | m-mul | l-mul | C-L |
:--: | :------: | :--: | :--: | :-: | :-: | :-: | :---: | :----: | :-------: | :---: | :---: | :---: | :-:
F | R-50 | L1Loss | 1x | N | N | 7.1 | 91.6 | 50.6 | 50.6 | 16.3 | 50.2 | 53.8
F | R-50 | IOULoss | 1x | N | N | 6.83 | 91.5 | 50.2 | 50.4 | 15.7 | 49.9 | 53.8
F | R-50 | GIOULoss | 1x | N | N | 6.76 | 91.5 | 49.5 | 50.2 | 16.0 | 49.8 | 53.7
F | R-50 | L1Loss | 1x | 0010 | N | 6.83 | 91.6 | 50.6 | 50.5 | 15.1 | 50.3 | 53.2
F | R-50 | L1Loss | 1x | 0010 | Y | 6.23 | 91.6 | 50.3 | 50.4 | 15.6 | 50.3 | 53.4
F | R-50 | L1Loss | 1x | 1111 | N | 5.30 | 91.6 | 50.0 | 50.4 | 15.0 | 49.9 | 53.8
F | R-50 | L1Loss | 1x | 1111 | Y | 5.14 | 91.6 | 51.6 | 51.2 | 16.3 | 50.6 | 54.8
F | R-101 | L1Loss | 1x | N | N | 5.59 | 91.6 | 51.1 | 50.8 | 15.1 | 50.2 | 54.7
F | R-101 | IOULoss | 1x | N | N | 5.68 | 91.5 | 49.5 | 50.1 | 14.5 | 49.6 | 53.6
F | R-101 | GIOULoss | 1x | N | N | 5.73 | 91.6 | 50.7 | 50.6 | 16.2 | 50.3 | 53.8
F | R-101 | L1Loss | 1x | 1111 | Y | 3.03 | 91.7 | 50.7 | 50.8 | 15.2 | 50.4 | 54.0
F | X-101 | L1Loss | 1x | N | N | 4.55 | 91.6 | 50.7 | 50.6 | 15.4 | 50.2 | 53.4
F | X-101 | IOULoss | 1x | N | N | 4.67 | 91.6 | 50.7 | 50.6 | 15.1 | 50.2 | 53.9
F | X-101 | GIOULoss | 1x | N | N | 4.67 | 91.6 | 50.5 | 50.3 | 16.4 | 49.7 | 54.1
L | R-50 | L1Loss | 1x | N | N | 6.40 | 91.4 | 51.9 | 51.2 | 14.6 | 51.0 | 53.9
L | R-50 | IOULoss | 1x | N | N | 6.40 | 92.1 | 50.2 | 50.5 | 14.7 | 50.1 | 53.6
L | R-50 | GIOULoss | 1x | N | N | 6.34 | 91.4 | 50.0 | 50.1 | 14.3 | 50.2 | 52.3
L | R-50 | L1Loss | 1x | 0010 | Y | 5.97 | 91.5 | 52.2 | 51.5 | 14.2 | 51.4 | 54.0
L | R-50 | L1Loss | 1x | 1111 | Y | 4.80 | 92.3 | 52.6 | 51.7 | 14.6 | 51.2 | 55.0
L | R-101 | L1Loss | 1x | N | N | 5.46 | 91.5 | 52.1 | 51.5 | 13.6 | 51.2 | 54.3
L | R-101 | IOULoss | 1x | N | N | 5.39 | 91.3 | 51.2 | 50.9 | 13.9 | 50.4 | 53.7
L | R-101 | GIOULoss | 1x | N | N | 5.46 | 91.4 | 49.4 | 50.2 | 13.5 | 49.9 | 52.9
L | X-101 | L1Loss | 1x | N | N | 4.58 | 92.2 | 51.9 | 51.6 | 13.7 | 51.1 | 54.3
L | X-101 | IOULoss | 1x | N | N | 4.58 | 91.4 | 50.9 | 50.7 | 14.1 | 50.4 | 53.4
L | X-101 | GIOULoss | 1x | N | N | 4.55 | 91.4 | 50.4 | 50.6 | 14.2 | 50.4 | 53.3
C | R-50 | L1Loss | 1x | N | N | 5.34 | 91.6 | 55.1 | 52.8 | 15.7 | 52.3 | 56.4
C | R-50 | IOULoss | 1x | N | N | 5.77 | 91.5 | 52.1 | 51.2 | 16.1 | 50.7 | 54.7
C | R-50 | GIOULoss | 1x | N | N | 3.84 | 91.5 | 52.6 | 51.6 | 15.8 | 51.0 | 55.4
C | R-50 | L1Loss | 1x | 0010 | N | 6.07 | 91.6 | 54.9 | 52.9 | 15.4 | 52.7 | 56.3
C | R-50 | L1Loss | 1x | 0010 | Y | 5.50 | 91.7 | 54.9 | 52.8 | 14.6 | 52.4 | 56.1
C | R-50 | L1Loss | 1x | 1111 | N | 4.90 | 91.6 | 54.7 | 52.8 | 16.3 | 52.2 | 56.3
C | R-50 | L1Loss | 1x | 1111 | Y | 4.70 | 91.7 | 55.1 | 52.9 | 15.4 | 52.5 | 56.5
C | R-101 | L1Loss | 1x | N | N | 4.52 | 91.6 | 55.0 | 52.8 | 15.1 | 52.4 | 55.6
C | R-101 | IOULoss | 1x | N | N | 5.04 | 91.6 | 52.5 | 51.4 | 15.4 | 50.8 | 55.2
C | R-101 | GIOULoss | 1x | N | N | 5.14 | 91.5 | 51.0 | 50.9 | 13.9 | 50.3 | 55.0
C | X-101 | L1Loss | 1x | N | N | 4.10 | 91.6 | 55.0 | 53.0 | 15.7 | 52.4 | 56.4
C | X-101 | IOULoss | 1x | N | N | 4.28 | 92.4 | 52.7 | 52.0 | 15.3 | 51.4 | 55.7
C | X-101 | GIOULoss | 1x | N | N | 4.36 | 91.6 | 52.3 | 51.4 | 16.2 | 50.8 | 55.4
V | R-50 | IOULoss | 1x | N | N | 6.45 | 93.4 | 56.5 | 54.4 | 17.4 | 54.0 | 57.8
V | R-50 | GIOULoss | 1x | N | N | 6.51 | 93.3 | 56.5 | 54.4 | 16.8 | 54.1 | 57.9
V | R-50 | IOULoss | 1x | 1111 | Y | 4.80 | 93.4 | 56.5 | 54.4 | 15.0 | 54.0 | 58.4
V | R-50 | GIOULoss | 1x | 1111 | Y | 4.90 | 92.9 | 56.7 | 54.6 | 16.3 | 54.2 | 58.2
V | R-101 | IOULoss | 1x | N | N | 5.30 | 93.6 | 56.6 | 54.6 | 16.9 | 54.1 | 58.5
V | R-101 | GIOULoss | 1x | N | N | 5.30 | 93.4 | 56.5 | 54.5 | 16.2 | 54.0 | 58.4
V | X-101 | IOULoss | 1x | N | N | 4.52 | 93.7 | 57.1 | 54.9 | 16.5 | 54.6 | 58.3
V | X-101 | GIOULoss | 1x | N | N | 5.39 | 93.6 | 56.9 | 54.8 | 16.2 | 54.2 | 58.5
* Our results are test in P100.
* Task: task network, contains Faster R-CNN(F), Libra R-CNN(L), Cascade R-CNN(C) and VFNet(V).
* Backbone: contains ResNet50, ResNet101 and ResNeXt101.
* Loss: contains L1Loss, IOULoss and GIOULoss.
* schd: contains 1x and 2x.
* **a** represents all with maxDets value 1000. **a-100** represents all with maxDets value 100. **mul** represent 0.5:0.95.
* C-L represent config and log files.
# Results and Logs (AR)
Task | Backbone | Loss | schd | Att | DCN | a-100 | a-300 | a-1000 | s-1000 | m-1000 | l-1000 |
:--: | :------: | :--: | :--: | :-: | :-: | :---: | :---: | :----: | :----: | :----: | :----: |
F | R-50 | L1Loss | 1x | N | N | 56.8 | 56.8 | 56.8 | 23.5 | 56.4 | 59.9
F | R-50 | IOULoss | 1x | N | N | 56.7 | 56.7 | 56.7 | 23.3 | 56.3 | 59.9
F | R-50 | GIOULoss | 1x | N | N | 56.4 | 56.4 | 56.4 | 20.8 | 56.0 | 59.7
F | R-50 | L1Loss | 1x | 0010 | N | 56.9 | 56.9 | 56.9 | 23.2 | 56.8 | 59.2
F | R-50 | L1Loss | 1x | 0010 | Y | 56.7 | 56.7 | 56.7 | 21.6 | 56.3 | 60.1
F | R-50 | L1Loss | 1x | 1111 | N | 56.9 | 56.9 | 56.9 | 24.3 | 56.5 | 60.1
F | R-50 | L1Loss | 1x | 1111 | Y | 57.1 | 57.1 | 57.1 | 23.5 | 56.6 | 60.8
F | R-101 | L1Loss | 1x | N | N | 56.8 | 56.8 | 56.8 | 23.5 | 56.2 | 60.9
F | R-101 | IOULoss | 1x | N | N | 56.3 | 56.3 | 56.3 | 20.3 | 55.8 | 60.1
F | R-101 | GIOULoss | 1x | N | N | 56.8 | 56.8 | 56.8 | 22.7 | 56.4 | 60.2
F | R-101 | L1Loss | 1x | 1111 | Y | 56.8 | 56.8 | 56.8 | 22.0 | 56.4 | 60.1
F | X-101 | L1Loss | 1x | N | N | 56.7 | 56.7 | 56.7 | 23.6 | 56.5 | 59.3
F | X-101 | IOULoss | 1x | N | N | 56.8 | 56.8 | 56.8 | 21.9 | 56.4 | 60.1
F | X-101 | GIOULoss | 1x | N | N | 56.4 | 56.4 | 56.4 | 22.0 | 55.8 | 60.4
L | R-50 | L1Loss | 1x | N | N | 57.6 | 57.6 | 57.6 | 26.4 | 57.5 | 59.7
L | R-50 | IOULoss | 1x | N | N | 57.2 | 57.2 | 57.2 | 25.0 | 56.9 | 59.8
L | R-50 | GIOULoss | 1x | N | N | 56.7 | 56.7 | 56.7 | 25.3 | 56.8 | 58.1
L | R-50 | L1Loss | 1x | 0010 | Y | 57.8 | 57.8 | 57.8 | 26.9 | 57.7 | 59.7
L | R-50 | L1Loss | 1x | 1111 | Y | 57.9 | 57.9 | 57.9 | 24.8 | 57.5 | 60.9
L | R-101 | L1Loss | 1x | N | N | 57.8 | 57.8 | 57.8 | 25.6 | 57.6 | 60.3
L | R-101 | IOULoss | 1x | N | N | 57.2 | 57.2 | 57.2 | 25.7 | 56.9 | 60.1
L | R-101 | GIOULoss | 1x | N | N | 56.6 | 56.6 | 56.6 | 24.9 | 56.4 | 58.9
L | X-101 | L1Loss | 1x | N | N | 57.9 | 57.9 | 57.9 | 27.0 | 57.7 | 60.2
L | X-101 | IOULoss | 1x | N | N | 57.3 | 57.3 | 57.3 | 26.0 | 57.1 | 59.7
L | X-101 | GIOULoss | 1x | N | N | 57.2 | 57.2 | 57.2 | 25.4 | 57.1 | 59.3
C | R-50 | L1Loss | 1x | N | N | 58.6 | 58.6 | 58.6 | 25.0 | 58.2 | 61.7
C | R-50 | IOULoss | 1x | N | N | 57.3 | 57.3 | 57.3 | 25.2 | 57.1 | 59.5
C | R-50 | GIOULoss | 1x | N | N | 57.6 | 57.6 | 57.6 | 25.5 | 57.3 | 60.6
C | R-50 | L1Loss | 1x | 0010 | N | 58.8 | 58.8 | 58.8 | 24.1 | 58.6 | 61.5
C | R-50 | L1Loss | 1x | 0010 | Y| 58.6 | 58.6 | 58.6 | 24.1 | 58.3 | 61.2
C | R-50 | L1Loss | 1x | 1111 | N | 58.6 | 58.6 | 58.6 | 26.1 | 58.3 | 61.5
C | R-50 | L1Loss | 1x | 1111 | Y | 58.7 | 58.7 | 58.7 | 26.2 | 58.4 | 61.8
C | R-101 | L1Loss | 1x | N | N | 58.6 | 58.6 | 58.6 | 24.0 | 58.4 | 60.8
C | R-101 | IOULoss | 1x | N | N | 57.4 | 57.4 | 57.4 | 23.3 | 57.2 | 60.0
C | R-101 | GIOULoss | 1x | N | N | 57.0 | 57.0 | 57.0 | 23.9 | 56.7 | 60.0
C | X-101 | L1Loss | 1x | N | N | 58.7 | 58.7 | 58.7 | 24.9 | 58.4 | 61.4
C | X-101 | IOULoss | 1x | N | N | 57.9 | 57.9 | 57.9 | 24.3 | 57.6 | 60.7
C | X-101 | GIOULoss | 1x | N | N | 57.5 | 57.5 | 57.5 | 24.8 | 57.1 | 60.4
V | R-50 | IOULoss | 1x | N | N | 61.1 | 61.1 | 61.1 | 25.7 | 60.6 | 64.9
V | R-50 | GIOULoss | 1x | N | N | 61.1 | 61.1 | 61.1 | 23.8 | 60.5 | 65.2
V | R-50 | IOULoss | 1x | 1111 | Y | 61.2 | 61.2 | 61.2 | 22.5 | 60.5 | 66.1
V | R-50 | GIOULoss | 1x | 1111 | Y | 61.2 | 61.2 | 61.2 | 24.8 | 60.6 | 65.4
V | R-101 | IOULoss | 1x | N | N | 61.2 | 61.2 | 61.2 | 24.4 | 60.6 | 65.5
V | R-101 | GIOULoss | 1x | N | N | 61.1 | 61.1 | 61.1 | 23.6 | 60.5 | 65.5
V | X-101 | IOULoss | 1x | N | N | 61.4 | 61.4 | 61.4 | 23.1 | 60.9 | 65.3
V | X-101 | GIOULoss | 1x | N | N | 61.2 | 61.2 | 61.2 | 22.0 | 60.6 | 65.8
* **a-100** represents all with maxDets value 100 (0.5:0.95)
# Postscript
* If you want to modify the related display effects of the detection box, such as the color of the detection box, the thickness of the detection box, etc., you can modify the **show_result** method in **/mmdet/models/detectors/base.py**. For details, please refer to this [document](https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/models/detectors/base.html?highlight=imshow_det_bboxes#). Pay attention to re-execute **pip install -v -e .** command after modification.
* When we train **Cascade-R-CNN-ResNeXt101**, the loss value is nan. The solution to this problem can be [referred to](https://github.com/open-mmlab/mmdetection/issues/3013). Specifically, add the gradient clip option in **cascade_rcnn_x101_32x4d_fpn_1x.py**, that is, add the following line of code **optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))**
