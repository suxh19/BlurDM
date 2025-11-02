# BlurDM: A Blur Diffusion Model for Image Deblurring (NeurIPS 2025)
Jin-Ting He, Fu-Jen Tsai, Yan-Tsung Peng, Min-Hung Chen, Chia-Wen Lin, Yen-Yu Lin
![Pipeline](assets/BlurDM_teaser.png)

## Installation
```
conda create -n IDBlau python=3.9
conda activate IDBlau
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv-python tqdm tensorboardX pyiqa thop numpy pandas
```

## Training
For each backbone, run the training scripts **in order**:

1. `train_stage1.py`
2. `train_stage2.py`
3. `train_stage3.py`

**Notes:**
- In `train_stage2.py`, load the **Encoder weights** trained in **Stage 1**.
- In `train_stage3.py`, load both:
  - the **deblurring backbone weights** pretrained in **Stage 1**, and
  - the **BlurDM weights** trained in **Stage 2**.

## Testing
You can either load your **trained weights** or download **our trained weights for each backbone** from [this link](<https://drive.google.com/drive/folders/144ntonNrjf_rjiDJQduzi9_5XhHI2TB-?usp=sharing>).  
Then run `deblur_predict.py` to evaluate on the test sets of each dataset.

**Important:**
- In `deblur_predict.py`, set the correct **path to the trained weights**.
- Also set the **dataset paths** before running.