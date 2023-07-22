from cvgl_data.backend import TiledWebMapsLoader, FrameLoader, Data, Loader, NamedData, NamedDataLoader

def getattr(self, name):
    if name in self.keys():
        return self[name]
    else:
        raise AttributeError(f"Object of type '{self.__class__}' has no attribute '{name}'")
setattr(NamedData, "__getattr__", getattr)
setattr(NamedDataLoader, "__getattr__", getattr)
setattr(NamedDataLoader, "__iter__", lambda self: iter(self.keys()))


from cvgl_data.backend import camera as backend_camera
class camera:
    @staticmethod
    def cam_op(x, args, kwargs, op):
        if isinstance(x, FrameLoader):
            op(x, *args, **kwargs)
        elif isinstance(x, list):
            for y in x:
                camera.cam_op(y, args, kwargs, op=op)
        else:
            raise ValueError(f"Invalid input type {type(x)}")

    @staticmethod
    def tile(x, *args, **kwargs):
        camera.cam_op(x, args, kwargs, op=backend_camera.tile)

    @staticmethod
    def resize_to_focal_length(x, *args, **kwargs):
        camera.cam_op(x, args, kwargs, op=backend_camera.resize_to_focal_length)

    @staticmethod
    def resize_by(x, *args, **kwargs):
        camera.cam_op(x, args, kwargs, op=backend_camera.resize_by)

    @staticmethod
    def constant_homography(x, *args, **kwargs):
        camera.cam_op(x, args, kwargs, op=backend_camera.constant_homography)

    @staticmethod
    def align_with_up_vector_homography(x, *args, **kwargs):
        camera.cam_op(x, args, kwargs, op=backend_camera.align_with_up_vector_homography)


from .util import *
from . import prepare, visualize
from .paired import bev_to_aerial
