# ©, 2019, Sirris
import os
import logging
from pathlib import Path, PosixPath
import requests
from tqdm import tqdm
logger = logging.getLogger(__name__)


class File(object):
    """
    Container to access a file stored on disk and download or generate it if it is not
    present.  To use, call the `make()` function.  If the file exists,
    its path will be returned.  If not, it will be computed supplied function.  
    This class is intended to access files
    that we generate.   It will make code more reusable as when others run your code, they
    will generate any files that are missing on their computer.
    It is a lower-level class; you should normally use
    DataFrameFile, DownloadFile, DataFrameDownload, or PickledObject.

    :param generate_file_func Function which generates the file.  Must have one
    arguemnt which is the path where the file is written.
    :param local_path Path where the file lives on disk.  It will be
    written to this path if it does not exist.

    """
    def __init__(self, generate_file_func, local_path):
        self.generate_func = generate_file_func
        self.local_path = Path(local_path)

    def _makedir(self):
        local_folder = self.local_path.parent
        if not local_folder.exists():
            logger.info("Creating directory " + str(local_folder))
            local_folder.mkdir()

    def _generate_file(self):
        self.clean()
        self._makedir()
        logger.info("Generating " + str(self.local_path))
        self.generate_func(self.local_path)

    def make(self, force=False):
        """Return the path of the file.  If it exists on disk, the path
        is returned directly.  If not, it will be computed using the supplied function.

        :param force (default False): If True, local 
        copies will be deleted, and the file will be regenerated.

        :returns
            File path (string).
        """

        if force or (not self.local_path.exists()):
            self._generate_file()
        
        logging.info("Using file: " + str(self.local_path))
        return(self.local_path)

    def clean(self):
        """Delete the preprocessed and the raw file."""
        if self.local_path.exists():
            logger.info("Deleting " + str(self.local_path))
            self.local_path.unlink()


def download_from_url(url, fname, force=False):
    """download file from url to target location. 
    From: https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads


    :param url: str. Url of file to download
    :param fname: str. Path to store file locally
    :param force: bool. Whether to force the download even if file exist. 
    Default: False

    :returns: fname.
    """
    fname = Path(fname) if not isinstance(fname, PosixPath) else fname
    if (not os.path.exists(fname)) | (force):
        # 
        if fname.parent.exists() is False:
            print(f'{fname.parent} directory does not exist and will be created')
            fname.parent.mkdir()
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        # open(fname, 'wb').close()
        with open(fname, 'wb') as file, tqdm(
            desc=f'Downloading file {fname}',
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    return fname


def download_from_repo(starterkit, dataset, fname, force=False):
    """download file from the datasets repo on github to target location. 


    :param starterkit: str. StarterKit ID (e.g. D3.4)
    :param dataset: str. Name of the dataset
    :param fname: str. Path to store file locally
    :param force: bool. Whether to force the download even if file exist. 
    Default: False

    :returns: fname.
    """

    url = ('https://github.com/EluciDATALab/elucidatalab.datasets/blob/main/data/'
           f'{starterkit}/{dataset}?raw=true')
    
    return download_from_url(url, fname, force=force)