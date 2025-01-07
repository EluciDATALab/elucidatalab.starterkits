from setuptools import find_packages, setup

setup(
    name='shadows',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    version='1.0',
    description='Tools for shadow compensation',
    author='Estephan Rustom',
    install_requires=['fastprogress', 'numpy', 'opencv_python', 'pandas', 
                      'Pillow', 'rasterio', 'scipy', 'torch', 'torchvision', 'scikit-image']
)
