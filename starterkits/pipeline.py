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
            local_folder.mkdir(parents=True)

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
            fname.parent.mkdir(parents=True)
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

dropbox_link_mapping = {
    'SK_1_1_1': {'original_files.tgz':
                    'https://www.dropbox.com/scl/fi/l7otey7yeh1s1iwur4dvq/original_files.tgz?rlkey=vya4212xkb610jyr8848kj4b2&st=a9tdp842&dl=1'},
    'SK_1_2_1': {'PM_data.zip':
                    'https://www.dropbox.com/scl/fi/62wcjdvcg3pr6b6l6k9a9/PM_data.zip?rlkey=em0wfi44vfm3qks0rm2yav3u8&st=6goj97f3&dl=1'},
    'SK_1_3': {'hourly_temperature.csv.zip':
                    'https://www.dropbox.com/scl/fi/eivn6xkkzpydoicpmxlta/hourly_temperature.csv.zip?rlkey=lke2e47oj0uhinztmks99bt09&st=bmeylhf6&dl=1',
               'household_power_consumption.txt.zip':
                    'https://www.dropbox.com/scl/fi/795rwptbqsmxm4r13fka0/household_power_consumption.txt.zip?rlkey=y71c6iez9gun7w04vvbzn2v2y&st=udf5u3pl&dl=1'},
    'SK_3_1': {'Seattle_bike_data.csv':
                    'https://www.dropbox.com/scl/fi/2rmggbgfgjq9g4jtjbkqv/Seattle_bike_data.csv?rlkey=pv89euwib4ewahmlntwlx1dyd&st=36g7fs06&dl=1',
               'Seattle_bike_data.csv.zip':
                    'https://www.dropbox.com/scl/fi/ec3t0asfer6d0ojqagnzt/Seattle_bike_data.csv.zip?rlkey=rs4aegfh13xe6jnc9tb67bhfg&st=wrfr0hph&dl=1'},
    'SK_3_2': {'SK-3-2-Pronto.zip':
                    'https://www.dropbox.com/scl/fi/9xulfp7iru0i2v5k1yyyz/SK-3-2-Pronto.zip?rlkey=l238kj0d4b34mroon2an37aox&st=l9w2cg9z&dl=1'},
    'SK_3_3': {'otp.csv.zip':
                    'https://www.dropbox.com/scl/fi/4hzetwk4apg9hqhx91osr/otp.csv.zip?rlkey=5vsazc44p344tjut8kmr0oaj9&st=35o8m335&dl=1',
               'trainView.csv.zip':
                    'https://www.dropbox.com/scl/fi/jjmz4vimeqp39kxanckdj/trainView.csv.zip?rlkey=8xd0mfzcnds4jakkcie6fbhuf&st=cp8cyr6v&dl=1'},
    'SK_3_4': {'la-haute-borne-data-2013-2016.zip':
                    'https://www.dropbox.com/scl/fi/cepl5vzvpr7u7htmf2lp0/la-haute-borne-data-2013-2016.zip?rlkey=2nsngcn7kw0m69desu86qqo2m&st=o9h1gmzh&dl=1'},
    'SK_4_1': {'IntelligentDataRetention.zip':
                    'https://www.dropbox.com/scl/fi/mctw2ug428ik9fcb8i7al/IntelligentDataRetention.zip?rlkey=wlzjb6ke7sn3kfeweckcmm28y&st=u21zr7dr&dl=1'},
    'SK_4_2': {'household_energy_consumption.zip':
                    'https://www.dropbox.com/scl/fi/al4opaos2pwo5seunb39f/household_energy_consumption.zip?rlkey=ji1ntyoqfuvbix9w2vm5eopu1&st=jwxmdttn&dl=1'},
}
def download_from_repo(starterkit, dataset, fname, force=False):
    """download file from the datasets repo on dropbox to target location.


    :param starterkit: str. StarterKit ID (e.g. D3.4)
    :param dataset: str. Name of the dataset
    :param fname: str. Path to store file locally
    :param force: bool. Whether to force the download even if file exist.
    Default: False

    :returns: fname.
    """

    # url = ('https://github.com/EluciDATALab/elucidatalab.datasets/blob/main/data/'
    #        f'{starterkit}/{dataset}?raw=true')
    # get link from dropbox
    url = dropbox_link_mapping[starterkit][dataset]
    return download_from_url(url, fname, force=force)
