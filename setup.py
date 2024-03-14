import os
import sys

from re import search
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
device = os.environ.get("DEVICE")
if not device:
    print('No device specified. Please set the DEVICE environment variable to '
          'either "cpu", "macos" or a cuda version (ie "cu118").')
    sys.exit(1)

if search('cu[0-9]{3}', device):
    device = device
    # TODO
    # code = subprocess.call(['pip', 'install', 'torch===1.13.1+', 'torchvision===0.8.1+cpu', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
elif search('cpu', device):
    device = 'cpu'
elif search('macos', device):
    device = 'darwin'
else:
    raise ValueError("Invalid device: {}. Must be either 'cpu', 'macos' or "
                     "'cuXXX'.".format(device))

with open('requirements.txt') as f:
    required_dependencies = f.read().splitlines()
    external_dependencies = []
    torch_added = False
    for dependency in required_dependencies:
        external_dependencies.append(dependency)

setup(
    name='TractOracleNet',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1',

    description='TractOracleNet',
    long_description="",

    # The project's main homepage.
    url='https://github.com/scil-vital/TractOracleNet',

    # Author details
    author='Antoine Théberge',
    author_email='antoine.theberge@usherbrooke.ca',

    # Choose your license
    license='GNU General Public License v3.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='Tractography',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['TractOracleNet'],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=external_dependencies,

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            "predictor.py=TractOracleNet.runners.predictor:main"]
    },
    include_package_data=True,

)
