from setuptools import setup, find_packages

setup(
    name='scepter',
    version='0.1',
    description='A Python package for scepter',
    author='Harry Qiu',
    author_email='hqiu678@outlook.com',
    url='https://github.com/hqiu-nju/scepter',
    packages=find_packages('scepter', exclude=('tests')),
    install_requires=[
        # Add your dependencies here
        'astropy',
        'pycraf',
        'cysgp4',
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
)