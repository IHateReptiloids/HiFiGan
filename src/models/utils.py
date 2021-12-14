from itertools import tee


def init_conv(conv_layer, mean=0.0, std=0.01):
    conv_layer.weight.data.normal_(mean, std)


def pairwise(iterable):
    '''
    there is no itertools.pairwise in python 3.7.10
    '''
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
