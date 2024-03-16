from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from model import (resnet, resnet_kgl,
                   alexnet, alexnet_kgl, 
                   vgg, vgg_kgl,
                   senet, senet_kgl, 
                   densenet, densenet_kgl,
                   simplenetv1, simplenetv1_kgl, 
                   efficientnetv2, efficientnetv2_kgl,
                   googlenet, googlenet_kgl, 
                   xception, xception_kgl,
                   mobilenetv2, mobilenetv2_kgl, 
                   inceptionv3, inceptionv3_kgl,
                   shufflenetv2, shufflenetv2_kgl, 
                   squeezenet, squeezenet_kgl,
                   mnasnet, mnasnet_kgl,
                   regnet, regnet_kgl,
                   convnext, convnext_kgl)


def load_model(model_name, in_channels=3, num_classes=10, **kwargs):
    print('\n[INFO] load model:', model_name)
    
    mu_threshold = kwargs.pop("mu_threshold", None)
    print('\n[INFO] mu_threshold:', mu_threshold)
    
    model = None
    if model_name == 'resnet50':
        if mu_threshold is None:
            model = resnet.resnet50(in_channels=in_channels, num_classes=num_classes)
        else:
            model = resnet_kgl.resnet50(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'resnext50':
        if mu_threshold is None:
            model = resnet.resnext50_32x4d(in_channels=in_channels, num_classes=num_classes)
        else:
            model = resnet_kgl.resnext50_32x4d(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'wide_resnet50':
        if mu_threshold is None:
            model = resnet.wide_resnet50_2(in_channels=in_channels, num_classes=num_classes)
        else:
            model = resnet_kgl.wide_resnet50_2(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'alexnet':
        if mu_threshold is None:
            model = alexnet.alexnet(in_channels=in_channels, num_classes=num_classes)
        else:
            model = alexnet_kgl.alexnet(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'vgg16':
        if mu_threshold is None:
            model = vgg.vgg16_bn(in_channels=in_channels, num_classes=num_classes)
        else:
            model = vgg_kgl.vgg16_bn(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'senet34':
        if mu_threshold is None:
            model = senet.seresnet34(in_channels=in_channels, num_classes=num_classes)
        else:
            model = senet_kgl.seresnet34(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'densenet121':
        if mu_threshold is None:
            model = densenet.densenet121(in_channels=in_channels, num_classes=num_classes)
        else:
            model = densenet_kgl.densenet121(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'simplenetv1':
        if mu_threshold is None:
            model = simplenetv1.simplenet(in_channels=in_channels, num_classes=num_classes)
        else:
            model = simplenetv1_kgl.simplenet(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'efficientnetv2s':
        if mu_threshold is None:
            model = efficientnetv2.effnetv2_s(in_channels=in_channels, num_classes=num_classes)
        else:
            model = efficientnetv2_kgl.effnetv2_s(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'googlenet':
        if mu_threshold is None:
            model = googlenet.googlenet(in_channels=in_channels, num_classes=num_classes)
        else:
            model = googlenet_kgl.googlenet(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'xception':
        if mu_threshold is None:
            model = xception.xception(in_channels=in_channels, num_classes=num_classes)
        else:
            model = xception_kgl.xception(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'mobilenetv2':
        if mu_threshold is None:
            model = mobilenetv2.mobilenetv2(in_channels=in_channels, num_classes=num_classes)
        else:
            model = mobilenetv2_kgl.mobilenetv2(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'inceptionv3':
        if mu_threshold is None:
            model = inceptionv3.inceptionv3(in_channels=in_channels, num_classes=num_classes)
        else:
            model = inceptionv3_kgl.inceptionv3(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'shufflenetv2':
        if mu_threshold is None:
            model = shufflenetv2.shufflenetv2(in_channels=in_channels, num_classes=num_classes)
        else:
            model = shufflenetv2_kgl.shufflenetv2(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'squeezenet':
        if mu_threshold is None:
            model = squeezenet.squeezenet(in_channels=in_channels, num_classes=num_classes)
        else:
            model = squeezenet_kgl.squeezenet(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold) 
    elif model_name == 'mnasnet':
        if mu_threshold is None:
            model = mnasnet.mnasnet(in_channels=in_channels, num_classes=num_classes)
        else:
            model = mnasnet_kgl.mnasnet(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'regnet_y_400mf':
        if mu_threshold is None:
            model = regnet.regnet_y_400mf(in_channels=in_channels, num_classes=num_classes)
        else:
            model = regnet_kgl.regnet_y_400mf(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    elif model_name == 'convnext_small':
        if mu_threshold is None:
            model = convnext.convnext_small(in_channels=in_channels, num_classes=num_classes)
        else:
            model = convnext_kgl.convnext_small(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)
    
    return model


if __name__ == "__main__":
    from torchsummary import summary
    model = load_model(model_name='convnext_small', in_channels=1, mu_threshold=0.3)

    print(model)

    summary(model, (1, 32, 32), device="cpu")