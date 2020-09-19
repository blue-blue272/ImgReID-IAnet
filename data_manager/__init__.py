from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .market1501_seg import Market1501_seg


__imgreid_factory = {
    'market1501_seg': Market1501_seg,
}



def get_names():
    return list(__imgreid_factory.keys())


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)
