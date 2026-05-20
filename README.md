# HazardFlow
Code Implementation of *HazardFlow: Score-based Energy Modeling for Health Risk Prediction*. 
This repository includes implementations of HazardFlow with multiple health risk prediction backbones, including **GRASP**, **ConCare**, **DDL-CXR**, **MedFuse**, and **DrFuse**. Other uni-modal backbones follow the same implementation pipeline to GRASP and ConCare.

Our implementation of score-based generative modeling is primarily built upon the tutorial code provided in https://github.com/jmtomczak/intro_dgm/blob/main/sbgms/sbgm_example.ipynb.


## GRASP
Our implementation builds upon *PyHealth* framework, and follows the original environment and usage guidelines provided at: https://github.com/sunlabuiuc/PyHealth/. 

We provide a revised version of PyHealth in this repository; installation follows the same steps as the original.

The implementation of GRASP with HazardFlow is located at:
```
pyhealth/models/sbs_grasp.py
```

You can run the model using:
```
python GRASP/grasp_hf.py
```

## ConCare
The implementation of ConCare with HazardFlow is located at:
```
pyhealth/models/sbs_concare.py
```

You can run the model using:
```
python ConCare/concare_hf.py
```

## DDL-CXR
Our implementation is built upon the original DDL-CXR codebase, and the environment setup follows the instructions provided in: https://github.com/Chenliu-svg/DDL-CXR.

To integrate HazardFlow with DDL-CXR, make the following modifications:
1. Replace
   ```
   DDL-CXR/ldm/models/predict_model.py
   ```
   with our version located at
   ```
   DDL-CXR-HF/ldm/models/predict_model.py
   ```
3. Add our configuration file
   ```
   DDL-CXR-HF/configs/Prediction/Pred_fusion_sbs.yaml
   ```
   to the directory
   ```
   DDL-CXR/configs/Prediction
   ```
The Python commands follow the original implementation pipeline.


## MedFuse
Our implementation is built upon the original MedFuse codebase, and the environment setup follows the instructions provided in: https://github.com/nyuad-cai/medfuse.

To integrate HazardFlow with MedFuse, please replace
```
MedFuse/models/fusion.py
```
with our version located at
```
MedFuse/fusion.py
```
and replace
```
MedFuse/models/fusion_trainer.py
```
with our version located at
```
MedFuse/fusion_trainer.py
```


## DrFuse
Our implementation is built upon the original DrFuse codebase, and the environment setup follows the instructions provided in: https://github.com/dorothy-yao/drfuse.

To integrate HazardFlow with DrFuse (we here provide the plug-and-play generalization version), please replace
```
DrFuse/main.py
```
with our version located at
```
DrFuse/main_twostage.py
```
replace
```
DrFuse/models/drfuse.py
```
with our version located at
```
DrFuse/drfuse.py
```
and replace
```
DrFuse/models/drfuse_trainer.py
```
with our version located at
```
DrFuse/drfuse_trainer.py
```
