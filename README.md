# wheat-detection
# Install Dependencies
```
conda create -n env_wheat
conda activate env_wheat
pip install torch==1.3.1
pip install cython
pip install torchvision==0.4.2
pip install albumentations
pip install imagecorruptions
pip install pycocotools
pip install terminaltables
pip install mmcv-full
sudo pip install -e .
```

If you run **pip install mmcv-full** meet wrong notification, you can see [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions. In our case, we use following command to successfully install mmcv.
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu100/torch1.3.1/index.html
```
# Prepare in advance

* You should first process the data into voc2017 format and put it in the **data** path.
* For subsequent training, you should modify the file under **config**. In our case, we have modified the following parts:
  * For **configs\_base_\models\faster_rcnn_r50_fpn.py** file
    * change num_classes=1 (line 46, 1 is represent the class nums);
  * For **configs\_base_\datasets\coco_detection.py** file
    * change data_root = 'data/wheat/' (line 2)
    * search **img_scale**, change it to (1024, 1024)
    * change **workers_per_gpu=0** (line 32)
    * change **samples_per_gpu=4** (line 31)
