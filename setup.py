#!/usr/bin/env python3

from setuptools import setup

setup(
    name="cvgl_data",
    version="0.1.1",
    description="A library for managing datasets for cross-view geolocalization (CVGL).",
    author="Florian Fervers",
    author_email="florian.fervers@gmail.com",
    packages=["cvgl_data"],
    package_data={"cvgl_data": ["*.so"]},
    license="MIT",
    install_requires=[
        "numpy",
        "pyarrow",
        "pyntcloud",
        "tqdm",
        "tinylogdir",
        "tiledwebmaps",
        "open3d",
        "docker",
        "pyquaternion",
        "imageio",
        "scikit-image",
        "opencv-python",
        "opencv-contrib-python",
        "pandas",
        "pyyaml",
        "nuscenes-devkit",
        "ciso8601",
        "pyunpack",
        "pypeln",
    ],
    url="https://github.com/fferflo/cvgl_data",
)
