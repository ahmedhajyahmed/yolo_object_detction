# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:15:04 2020

@author: ppike
"""


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('./object_detection/yolov4/requirements.txt')

# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir) for ir in install_reqs if not str(ir).startswith("-") ]

print('Requirements',reqs)

setuptools.setup(
    name="object_detection", # Replace with your own username
    version="0.0.1",
    author="Paul",
    #author_email="author@example.com",
    #description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[reqs]
)