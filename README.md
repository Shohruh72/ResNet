# ResNet
ResNet re-implementation using PyTorch
# Steps
** Configure imagenet path by changing data_dir in main.py
** bash ./main.sh $ --train for training model, $ is number of GPUs
** python main.py --test for testing
### Results

| Version   | Top@1 | Top@5 | Params | FLOPS | Pretrained weights |
|:---------:|:-----:|------:|-------:|------:|-------------------:|
| Resnet 18 |  71.5 | 89.9  | 11.7M  | 1.8G  |[model](https://github.com/Shohruh72/ResNet/releases/download/v0.0.1/resnet_18.pt)|
| Resnet 34 |  76.3 | 92.8  | 21.8M  | 3.6G  |[model](https://github.com/Shohruh72/ResNet/releases/download/v0.0.1/resnet_34.pt)|
| Resnet 50 |  80.1 | 94.5  | 25.5M  | 4.1G  |[model](https://github.com/Shohruh72/ResNet/releases/download/v0.0.1/resnet_50.pt)|
| Resnet 101 | 81.2 | 95.1  | 44.5M  | 7.8G  |[model](https://github.com/Shohruh72/ResNet/releases/download/v0.0.1/resnet_101.pt)|
| Resnet 152 | 81.7 | 95.1  | 60.1M  | 11.6G |[model](https://github.com/Shohruh72/ResNet/releases/download/v0.0.1/resnet_152.pt)|
