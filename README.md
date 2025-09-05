## Quanta SD 3.5 Large

### Stage 1:

> accelerate launch train_daEncoder_SD35.py --config configs/train/train_daEncoder.yaml

**Mosaic versions:**

> accelerate launch train_daEncoder_SD35.py --config configs/train/train_stage1_3-bit_mosaic.yaml
> accelerate launch --main_process_port 29501 train_daEncoder_SD35.py --config configs/train/train_stage1_1-bit_mosaic.yaml