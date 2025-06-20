import os

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision
import torch

from models.layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC, Rpool
from models.layers.normalization import L2Norm, PowerLaw


def get_root():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    )


def get_data_root():
    return os.path.join(get_root(), "data")


# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is higher for them
FEATURES = {
    "vgg16": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth",
    "resnet50": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth",
    "resnet101": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth",
    "resnet152": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth",
}

# TODO: pre-compute for more architectures and properly test variations (pre l2norm, post l2norm)
# pre-computed local pca whitening that can be applied before the pooling layer
L_WHITENING = {
    "resnet101": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-lwhiten-9f830ef.pth",  # no pre l2 norm
    # 'resnet101' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-lwhiten-da5c935.pth', # with pre l2 norm
}

# possible global pooling layers, each on of these can be made regional
POOLING = {
    "mac": MAC,
    "spoc": SPoC,
    "gem": GeM,
    "gemmp": GeMmp,
    "rmac": RMAC,
}

# TODO: pre-compute for: resnet50-gem-r, resnet50-mac-r, vgg16-mac-r, alexnet-mac-r
# pre-computed regional whitening, for most commonly used architectures and pooling methods
R_WHITENING = {
    "alexnet-gem-r": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-rwhiten-c8cf7e2.pth",
    "vgg16-gem-r": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-rwhiten-19b204e.pth",
    "resnet101-mac-r": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-rwhiten-7f1ed8c.pth",
    "resnet101-gem-r": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-rwhiten-adace84.pth",
}

# TODO: pre-compute for more architectures
# pre-computed final (global) whitening, for most commonly used architectures and pooling methods
WHITENING = {
    "alexnet-gem": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-whiten-454ad53.pth",
    "alexnet-gem-r": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-whiten-4c9126b.pth",
    "vgg16-gem": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-whiten-eaa6695.pth",
    "vgg16-gem-r": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-whiten-83582df.pth",
    "resnet50-gem": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet50-gem-whiten-f15da7b.pth",
    "resnet101-mac-r": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-whiten-9df41d3.pth",
    "resnet101-gem": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth",
    "resnet101-gem-r": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-whiten-b379c0a.pth",
    "resnet101-gemmp": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gemmp-whiten-770f53c.pth",
    "resnet152-gem": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet152-gem-whiten-abe7b93.pth",
    "densenet121-gem": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet121-gem-whiten-79e3eea.pth",
    "densenet169-gem": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet169-gem-whiten-6b2a76a.pth",
    "densenet201-gem": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet201-gem-whiten-22ea45c.pth",
}

# output dimensionality for supported architectures
OUTPUT_DIM = {
    "alexnet": 256,
    "vgg11": 512,
    "vgg13": 512,
    "vgg16": 512,
    "vgg19": 512,
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "densenet121": 1024,
    "densenet169": 1664,
    "densenet201": 1920,
    "densenet161": 2208,  # largest densenet
    "squeezenet1_0": 512,
    "squeezenet1_1": 512,
}


class ImageRetrievalNet(nn.Module):
    def __init__(self, features, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()

        self.backbone = nn.Sequential(*features)
        
        # Mean head
        self.mean_pool = nn.Sequential(*[pool, nn.Flatten()])
        self.mean_head = nn.Sequential(
            whiten,
            nn.Linear(meta["outputdim"], meta["outputdim"] - 1),
            L2Norm()
        )
        
        # Variance head
        self.var_pool = nn.Sequential(*[pool, nn.Flatten()])
        self.var_head = nn.Sequential(
            nn.Linear(meta["outputdim"], 1),
            nn.Softplus()
        )

        self.meta = meta

    def forward(self, x, n_samples=1):
        features = self.backbone(x)
        
        # Mean embeddings
        mean_pooled = self.mean_pool(features)
        mu = self.mean_head(mean_pooled)
        
        # Variance embeddings
        var_pooled = self.var_pool(features)
        sigma2 = self.var_head(var_pooled)
        
        # Ensure sigma2 has the same shape as mu
        sigma2 = sigma2.expand_as(mu)
        
        # Concatenate mean and variance
        z = torch.cat([mu, sigma2], dim=1)
        
        return {"z_mu": z}

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ")"
        return tmpstr

    def meta_repr(self):
        tmpstr = "  (" + "meta" + "): dict( \n"
        tmpstr += "     architecture: {}\n".format(self.meta["architecture"])
        tmpstr += "     pooling: {}\n".format(self.meta["pooling"])
        tmpstr += "     outputdim: {}\n".format(self.meta["outputdim"])
        tmpstr = tmpstr + "  )\n"
        return tmpstr


def init_network(params):
    # parse params with default values
    architecture = params.get("architecture", "resnet101")
    pooling = params.get("pooling", "gem")
    dropout_rate = params.get("dropout_rate", 0.0)
    use_mc_dropout = params.get("use_mc_dropout", False)

    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    # initialize with random weights
    net_in = getattr(torchvision.models, architecture)(pretrained=True)

    # initialize features
    if architecture.startswith("alexnet"):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith("vgg"):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith("resnet"):
        features = list(net_in.children())[:-2]
    elif architecture.startswith("densenet"):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith("squeezenet"):
        features = list(net_in.features.children())
    else:
        raise ValueError(
            "Unsupported or unknown architecture: {}!".format(architecture)
        )

    # Only add dropout if use_mc_dropout is True and dropout_rate > 0
    if use_mc_dropout and dropout_rate > 0:
        new_features = []
        for i, layer in enumerate(features):
            new_features.append(layer)
            if isinstance(layer, (nn.ReLU, nn.MaxPool2d)) and i < len(features) - 1:
                new_features.append(nn.Dropout2d(dropout_rate))
        features = new_features

    pool = POOLING[pooling]()

    # initialize whitening
    whiten = nn.Linear(dim, dim, bias=True)
    if use_mc_dropout and dropout_rate > 0:
        whiten = nn.Sequential(nn.Dropout(dropout_rate), whiten)

    meta = {
        "architecture": architecture,
        "pooling": pooling,
        "outputdim": dim,
    }

    net = ImageRetrievalNet(features, pool, whiten, meta)
    return net

