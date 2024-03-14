from setuptools import setup, find_packages
from setuptools import setup, find_packages

setup(
    name='your-package-name',
    version='0.1.0',
    author='Your Name',
    author_email='your@email.com',
    description='A short description of your package',
    packages=find_packages(),
    install_requires=[
        # Add any dependencies your package requires
        'numpy',
        'matplotlib',
        'astropy',
        'pycraf',
        'cysgp4',
        'python>=3.10',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    url='https://github.com/hqiu-nju/scepter',
    }
)