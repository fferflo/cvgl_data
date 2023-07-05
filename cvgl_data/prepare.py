import os, pyunpack, tqdm, imageio, requests, tarfile, cosy, cv2, skimage.transform, cvgl_data
import numpy as np
import pypeln as pl

image_extensions = ["jpg", "jpeg", "png", "tif", "tiff"]

def is_image_file(file):
    return any([file.endswith("." + ext) for ext in image_extensions])

def compute_bearings_for_latlons(latlons, min_distance_for_bearing=2.0): # meters
    bearings = []
    for i in range(len(latlons)):
        for d in range(len(latlons)):
            i0 = max(i - d, 0)
            i1 = min(i + d, len(latlons) - 1)
            if cosy.geo.distance(latlons[i0], latlons[i1]) > min_distance_for_bearing:
                bearings.append(cosy.geo.bearing(latlons[i0], latlons[i1]))
                break
        else:
            raise ValueError(f"Failed to compute bearing in trajectory from {latlons[0]} to {latlons[1]}, path length is below minimum distance")
    assert len(bearings) == len(latlons)
    return np.asarray(bearings)

def download(url, file, retries=100, timeout=10.0):
    dir = os.path.dirname(os.path.abspath(file))
    if not os.path.isdir(dir):
        os.makedirs(dir)

    for _ in range(retries):
        if os.path.isfile(file):
            os.remove(file)
        elif os.path.isdir(file):
            raise ValueError("Target path is a directory")
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            total = int(resp.headers.get("content-length", 0))
            received = 0
            with open(file, "wb") as f, tqdm.tqdm(desc="Download " + url.split("/")[-1], total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
                for data in resp.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
                    received += size
            if received < total:
                error = requests.exceptions.RequestException("Content too short", response=resp)
                continue
        except requests.exceptions.RequestException as e:
            error = e
            continue
        break
    else:
        if os.path.isfile(file):
            os.remove(file)
        raise error

def extract(src, dest=None):
    if dest is None:
        dest = os.path.dirname(src)
    print(f"Extracting {src}")
    if not os.path.isdir(dest):
        os.makedirs(dest)
    if src.endswith(".tar"):
        tar = tarfile.open(src)
        tar.extractall(path=dest)
        tar.close()
    else:
        pyunpack.Archive(src).extractall(dest)
    os.remove(src)

def resize(path, min_size, skip=lambda n: False, preprocess=None, load=imageio.imread):
    min_size = np.asarray(min_size)

    tempfile_marker = "__resize-images-temp-marker"
    def to_tempfile(file):
        file, extension = os.path.splitext(file)
        return file + tempfile_marker + extension
    def from_tempfile(file):
        file, extension = os.path.splitext(file)
        assert file.endswith(tempfile_marker)
        file = file[:-len(tempfile_marker)]
        return file + extension
    def is_tempfile(file):
        file, extension = os.path.splitext(file)
        return file.endswith(tempfile_marker)

    tasks = []
    for subpath, dirs, files in os.walk(path):
        for file in files:
            if is_image_file(file):
                file = os.path.join(subpath, file)
                if skip(file):
                    continue
                if is_tempfile(file):
                    tempfile = file
                    file = from_tempfile(file) # TODO: this file has the wrong extension
                    if os.path.isfile(file):
                        print(f"Found temp-file {tempfile}, removing")
                        os.remove(tempfile)
                    else:
                        print(f"Found temp-file {tempfile}, renaming to {os.path.basename(file)}")
                        os.rename(tempfile, file)
                else:
                    tasks.append(file)
    print(f"Found {len(tasks)} images in {path} that will be resized")

    stream = tasks

    # Load frame
    def load2(file):
        try:
            image = load(file)
        except ValueError as e:
            print(f"Failed to resize image file {file}")
            raise e

        if len(image.shape) == 3 and image.shape[2] >= 3 and (np.all(np.asarray(image.shape[:2]) > min_size) or not file.endswith(".jpg") or not preprocess is None):
            image = image[:, :, :3]
            return [(image, file)]
        else:
            return []
    stream = pl.thread.flat_map(load2, stream, workers=4, maxsize=4)

    # Convert frame
    @cvgl_data.unwrap
    def convert(image, file):
        if not preprocess is None:
            image = preprocess(image, file)

        shape = np.asarray(image.shape[:2])
        factor = np.amax(min_size.astype("float") / shape)
        if factor < 1.0:
            shape = (shape * factor).astype("int")

            dtype = image.dtype
            image = skimage.transform.resize(image.astype("float32"), shape, order=1, mode="constant", preserve_range=True, anti_aliasing=True)
            image = image.astype(dtype)

        return image, file
    stream = pl.thread.map(convert, stream, workers=12, maxsize=12)

    # Save frame
    @cvgl_data.unwrap
    def save(image, file_oldext):
        file_jpg = ".".join(file_oldext.split(".")[:-1]) + ".jpg"
        tempfile_jpg = to_tempfile(file_jpg)
        imageio.imwrite(tempfile_jpg, image)
        os.remove(file_oldext)
        os.rename(tempfile_jpg, file_jpg)
    stream = pl.thread.map(save, stream, workers=4, maxsize=4)

    for _ in tqdm.tqdm(stream, total=len(tasks)):
        pass

class RunException(BaseException):
    def __init__(self, message, code):
        self.message = message
        self.code = code

def run(command):
    print("> " + command)
    returncode = os.system(f"bash -c '{command}'")
    if returncode != 0:
        raise RunException("Failed to run " + command + ". Got return code " + str(returncode), returncode)

def pano_to_pinhole(image, shape, intr, theta, phi):
    def xyz2lonlat(xyz):
        norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
        xyz_norm = xyz / norm
        x = xyz_norm[..., 0:1]
        y = xyz_norm[..., 1:2]
        z = xyz_norm[..., 2:]

        lon = np.arctan2(x, z)
        lat = np.arcsin(y)
        lst = [lon, lat]

        out = np.concatenate(lst, axis=-1)
        return out

    def lonlat2XY(lonlat, shape):
        x = (lonlat[..., :1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
        y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
        out = np.concatenate([x, y], axis=-1)

        return out

    x = np.arange(shape[1])
    y = np.arange(shape[0])
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=-1)
    xyz = xyz @ np.linalg.inv(intr).T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * theta)
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * phi)
    R = R2 @ R1
    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz)
    xy = lonlat2XY(lonlat, shape=image.shape[:2]).astype(np.float32)
    pinhole = cv2.remap(image, xy[..., 0], xy[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

    return pinhole
