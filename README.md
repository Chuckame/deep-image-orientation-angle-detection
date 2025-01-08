# Deep Image Orientation Angle Detection

## Introduction
Image Orientation Angle Detection Model (Deep-OAD) is a deep learning model to predict the orientation angle of the natural images. This repository allows to train, finetune and predict with a Deep-OAD model. The academic paper is available here:  https://arxiv.org/abs/2007.06709

## Saved Model
You can download a trained Keras model [here](https://drive.google.com/file/d/1sdmPmaDhivdHPfn9M9vAkTbiprbPq94e/view?usp=share_link). This model is trained on artifically created dataset using almost all of the images of Microsoft COCO. This model is capable to predict orientation of images between 0° to 359° with test MAE of 6.5°. You can use this model for finetuning or model predictions. Please make sure the model file is put under [weights](./weights/) directory during finetuning or inference.

## Model Inference
To predict orientation angle of an image from terminal run the following command,

```shell
python3.11 -m venv .venv
source ./.venv/bin/activate
python3.11 -m pip install -r requirements.txt
python3.11 ./infer-rotation.py --image-path ./inputs/download.jpeg
```

## Convert weights to onnx

- save model using SavedModel format
- use tf2onnx


## Citation
This paper is submitted for journal publication. If you are using this model then please use the below BibTeX to cite for now.

```
@ARTICLE{2020arXiv200706709M,
       author = {{Maji}, Subhadip and {Bose}, Smarajit},
        title = "{Deep Image Orientation Angle Detection}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning, Electrical Engineering and Systems Science - Image and Video Processing},
         year = 2020,
        month = jun,
          eid = {arXiv:2007.06709},
        pages = {arXiv:2007.06709},
archivePrefix = {arXiv},
       eprint = {2007.06709},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200706709M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## More Information
For any clarification feel free to raise an issue. Additionally you can reach us at subhadipmaji.jumech@gmail.com
