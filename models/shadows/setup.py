from setuptools import find_packages, setup

setup(
    name='shadows',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    version='1.0',
    description='Tools for shadow compensation',
    author='Estephan Rustom',
    install_requires=['fastprogress==1.0.3','matplotlib==3.7.5','numpy==1.24.4','opencv_python==4.10.0.84',
                      'pandas==2.0.3','Pillow==9.0.1','pvlib==0.11.0','pymeanshift==0.2.2','pyproj==3.5.0',
                      'rasterio==1.3.10','scikit_learn==1.3.2','scipy==1.10.1','setuptools==74.1.2',
                      'scikit-image==0.21.0','torch==2.3.1','torchvision==0.18.1','tqdm==4.62.3']
)
