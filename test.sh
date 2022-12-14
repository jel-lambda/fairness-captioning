# python test_COCO.py --network vgg19 --model model/model_vgg19_1.pth --use-filter
python test_fairCOCO.py --images-dir /home/ubuntu/efs/final/ --annotations-dir data/annotations/ --network vgg19\
                        --model model/model_vgg19_1.pth --use-filter