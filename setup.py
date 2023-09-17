from setuptools import setup, find_packages
from module_structure import __author__, __version__, __name__

AUTHOR = __author__
VERSION = __version__
NAME = __name__

setup(
    name=NAME,
    version=VERSION,
    description='Machine Learning Classes UFV',
    author=AUTHOR,
    author_email='manuel.mirandacortes@gmail.com',
    license='MIT',
    python_requires='>=3.11',
    packages=find_packages(),
    include_package_data=False,
    install_requieres= ['pandas', 'torch', 'numpy']
)