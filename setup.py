from setuptools import setup, find_packages

setup(
	name="starterkits",
	packages=find_packages(),
	version="1.0.1",
	description="A library of data science, ML and AI algorithms and code utilities",
	install_requires=['calmap', 'tensorflow', 'dash_bootstrap_components==0.13.1', 'jupyter_dash', ],
	author="EluciDATA Lab by Sirris",
	author_email="t-dat@sirris.be",
	url="https://github.com/EluciDATALab/elucidatalab.starterkits.git"
)
