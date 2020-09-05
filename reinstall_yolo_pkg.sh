#!/bin/bash

pip uninstall object_detection -y

python3 setup.py sdist bdist_wheel

pip install dist/object_detection-0.0.1-py3-none-any.whl
