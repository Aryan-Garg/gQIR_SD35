## Quanta SD 3.5 Large

### Stage 1:

> accelerate launch train_daEncoder_SD35.py --config configs/train/train_daEncoder.yaml

**Mosaic versions:**

> accelerate launch train_daEncoder_SD35.py --config configs/train/train_s1_mosaic_default_3bit.yaml
> accelerate launch --main_process_port 29501 train_daEncoder_SD35.py --config configs/train/train_stage1_1-bit_mosaic.yaml

### Stage 2:

>  CUDA_VISIBLE_DEVICES=3 python3 train_s2_SD35.py --config configs/train/train_s2_3bit.yaml
>  CUDA_VISIBLE_DEVICES=4 python3 train_s2_SD35.py --config configs/train/train_s2_3bit_monochrome.yaml