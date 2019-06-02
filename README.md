# Multi-Label Out-of-Distribution Detection

Code for mutli-label OOD detection experiments from [A Benchmark for Anomaly Segmentation](https://arxiv.org/abs/1911.11132).

### Requirements

Pytorch >= 0.4.1  

### Datasets

First download PASCAL VOC from [here](host.robots.ox.ac.uk/pascal/VOC/
) or from a popular [mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).

Download MS-COCO 2014 dataset from [here](http://cocodataset.org/#download).

For OOD experiments we use a subset of ImageNet-22K which can be downloaded from [here](https://drive.google.com/file/d/1ciBRbMpaN0FGgaFwjmjOlX3eeD8WrRjL/view?usp=sharing). The full ImageNet-22K can be downloaded from [here](http://image-net.org/download-images).

Install the pycocotools. We used the following command: 

    pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
    Alternative pycocotools installation
    # pip3 install git+https://github.com/waleedka/coco.git#egg=pycocotools&subdirectory=PythonAPI

We have parsed the PASCAL VOC labels and included them in the datasets folder.  For the dataset to work one should create a symbolic link called "Pascal" in the root directory that points to the location above VOCdevkit/


## Running the experiments for PASCAL VOC

Create the symlink to the location of Pascal dataset

    ln -s path/to/PASCALdataset Pascal

Note: within PASCALdataset folder should contain VOCdevkit

Train the pascal model

    python3 train.py  --dataset=pascal

Evaluate the model on PASCAL VOC

    python3 validate.py

Test the model for OOD experiments

    python3 eval_ood.py

## Running the experiments for MS-COCO

Preprocess the COCO dataset.

    python3 utils/coco-preprocessing.py  path/to/coco-dataset

Train the model on the COCO dataset

    python3 train.py --disc=cocomodel --dataset=coco

Evaluate the model on the COCO validation set

    python3 validate.py --disc=cocomodel --dataset=coco --split=multi-label-val2014

Finally run the tests for OOD on the coco model

    python3 eval_ood.py --disc=cocomodel --dataset=coco --split=multi-label-val2014

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2019anomalyseg,
      title={A Benchmark for Anomaly Segmentation},
      author={Hendrycks, Dan and Basart, Steven and Mazeika, Mantas and Mostajabi, Mohammadreza and Steinhardt, Jacob and Song, Dawn},
      journal={arXiv preprint arXiv:1911.11132},
      year={2019}
    }
