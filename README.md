# DeepLung:

## Zeding Edit:
## Start with some basic intro of methodlogies and knowledge to understand the code better
## What is 10-Fold Cross-Validation?
### Splitting the Data:

The dataset is divided into 10 equal parts, or "folds."
Each fold contains approximately the same number of samples.
### Training and Testing:

For each iteration, the model is trained on 9 folds and tested on the remaining 1 fold.
This process is repeated 10 times, with each fold being used as the test set exactly once.
### Performance Aggregation:

The model’s performance metrics (e.g., accuracy, precision, recall) are calculated for each fold.
The final result is the average performance across all 10 folds.

## Follow the below steps to run training and evaluation
1. Update config_training.py (for preprocessing), this config is only for data preprocessing, prepare.py
2. Update config_training9.py (for training), this config is for training and testing, need to updated according to 10-fold validation. If 10-folder validatin is needed, then we need to create 10 config file with different setting and repeat the entire training and testing process 10 times
2. Run data preprocessing: prepare.py, this steps only needs to be run one time, it performs data preprocessing based on the LUNA16 data.
3. Train the model: python main.py --model res18 -b 8 --resume 064.ckpt --save-dir res18/retrft969/ --epochs 150 --config config_training9, Model is trained on top of a pretrained model called 064.ckpt, which is based on another dataset(maybe tianchi). The model training is resumed from the pretrained model to increase the model performance. We can train from scratch based on LUNA16 by setting resume to empty. The performance may be worse. The training step will generate X number of models corresponding to Epoch#, saved under detector/results/modelname/... 
4. Test the model: run run_training.py This step will generate lbb(Ground Truth Bounding Boxes) and pbb(Predicted Bounding Boxes), which will be used to calculate FROC
5. Generate FROC result: run froc_step1.py
6. Generate FROC result: run froc_step2.py
7. Generate FROC result: run froc_step2.1.py




Please add paper into reference if the repository is helpful to you.

Zhu, Wentao, Chaochun Liu, Wei Fan, and Xiaohui Xie. "DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary Nodule Detection and Classification." IEEE WACV, 2018.

Deepem: Deep 3d convnets with em for weakly supervised pulmonary nodule detection, MICCAI, 2018.

Dependecies: Ubuntu 14.04, python 2.7, CUDA 8.0, cudnn 5.1, h5py (2.6.0), SimpleITK (0.10.0), numpy (1.11.3), nvidia-ml-py (7.352.0), matplotlib (2.0.0), scikit-image (0.12.3), scipy (0.18.1), pyparsing (2.1.4), pytorch (0.1.10+ac9245a) (anaconda is recommended)

Download LUNA16 dataset from https://luna16.grand-challenge.org/data/

Download LIDC-IDRI dataset from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

For preprocessing, run ./DeepLung/prepare.py. The parameters for prepare.py is in config_training.py.<br />
*_data_path is the unzip raw data path for LUNA16.<br />
*_preprocess_result_path is the save path for the preprocessing.<br />
*_annos_path is the path for annotations.<br />
*_segment is the path for LUNA16 segmentation, which can be downloaded from LUNA16 website.

Use run_training.sh to train the detector.<br />
You can use the ResNet or dual path net model by revising --model attribute.<br />
After training and test are done, use the ./evaluationScript/frocwrtdetpepchluna16.py to validate the epoch used for test.<br />
After that, collect all the 10 folds' prediction, use ./evaluationScript/noduleCADEvaluationLUNA16.py to get the FROC for all 10 folds.<br />
You can directly run noduleCADEvaluationLUNA16.py, and get the performance in the paper.

The trained model is in ./detector/dpnmodel/ or ./detector/resmodel/
The performances on each fold are (these results are in the supplement)

Method	Deep 3D Res18	Deep 3D DPN26

Fold 0	0.8610	      0.8750

Fold 1	0.8538	      0.8783

Fold 2	0.7902      	0.8170

Fold 3	0.7863      	0.7731

Fold 4	0.8795	      0.8850

Fold 5	0.8360  	    0.8095

Fold 6	0.8959  	    0.8649

Fold 7	0.8700      	0.8816

Fold 8	0.8886	      0.8668

Fold 9	0.8041    	  0.8122

The performances on each average false positives in FROC compared with other approaches (these results are in the supplement)

Methods         0.125 0.25  0.5 1 2 4 8 FROC

DIAG_ConvNet    0.692 0.771 0.809 0.863 0.895 0.914 0.923 0.838

ZENT            0.661 0.724 0.779 0.831 0.872 0.892 0.915 0.811

Aidence         0.601 0.712 0.783 0.845 0.885 0.908 0.917 0.807

MOT_M5Lv1       0.597 0.670 0.718 0.759 0.788 0.816 0.843 0.742

VisiaCTLung     0.577 0.644 0.697 0.739 0.769 0.788 0.793 0.715

Etrocad         0.250 0.522 0.651 0.752 0.811 0.856 0.887 0.676

Dou et al 2017  0.659 0.745 0.819 0.865 0.906 0.933 0.946 0.839

3D RES          0.662 0.746 0.815 0.864 0.902 0.918 0.932 0.834

3D DPN          0.692 0.769 0.824 0.865 0.893 0.917 0.933 0.842

For nodule classification, first clean the data from LIDC-IDRI.<br />
Use the ./data/extclsshpinfo.py to extract nodule labels.<br />
humanperformance.py is used to get the performance of doctors. 

dimcls.py is used to get the classification based on diameter.<br />
nodclsgbt.py is used to get the performance based on GBM, nodule diameter and nodule pixel.<br />
pthumanperformance.py is used for patient-level diagnosis performance.<br />
kappatest.py is used for kappa value calculation in the paper.

For classification using DPN, use the code in main_nodcls.py.<br />
You may revise the code a little bit for different test settings.

For system's classification, that is classification based on detection.<br />
First, use the detection's test script in the run_training.sh to get the detected nodules for training CTs.<br />
Use the det2cls.py to train the model. And use the testdet2cls.py to test the trained model.<br />
You may revise the code a little bit for different test settings.

Doctor's annotation for each nodule in LIDC-IDRI is in ./nodcls/annotationdetclssgm_doctor.csv

Feel free to ask any questions. Wentao Zhu, wentaozhu1991@gmail.com

LIDC-IDRI nodule size report downloaded from 
http://www.via.cornell.edu/lidc/list3.2.csv is in /nodcls/data/list3.2.csv
