# wheat-detection
# Install Dependencies
- conda create -n env_wheat
- conda activate env_wheat
- pip install torch==1.3.1
- pip install cython
- pip install torchvision==0.4.2
- pip install albumentations
- pip install imagecorruptions
- pip install pycocotools
- pip install terminaltables
- pip install mmcv-full
- sudo pip install -e .

If you run **pip install mmcv-full** meet wrong notification, you can see [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions. In our case, we use **[pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu100/torch1.3.1/index.html]** to successfully install mmcv.
