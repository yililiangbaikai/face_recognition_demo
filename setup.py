'''Setup script for object_detection with webrtc'''

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['Pillow>=1.0', 'Flask', 'tensorflow', 'six', 'matplotlib']

setup(
    name='webrtc_object_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages()],
    description='Tensorflow Object Detection with WebRTC',
)

'''Download the Object Dectection directory'''
import six.moves.urllib as urllib
from zipfile import ZipFile
import os
import re
import shutil

print("\n\nDownloading the TensorFlow API from Github...")

