from cvgl_data.backend import TiledWebMapsLoader, FrameLoader, Data, Loader, NamedData, NamedDataLoader, cam_ops

def getattr(self, name):
    if name in self.keys():
        return self[name]
    else:
        raise AttributeError(f"Object of type '{self.__class__}' has no attribute '{name}'")
setattr(NamedData, "__getattr__", getattr)
setattr(NamedDataLoader, "__getattr__", getattr)
setattr(NamedDataLoader, "__iter__", lambda self: iter(self.keys()))

from .util import *
from . import prepare, visualize
from .paired import bev_to_aerial
