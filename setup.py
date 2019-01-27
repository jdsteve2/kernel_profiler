#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

ver_dic = {}
version_file = open("kernel_profiler/version.py")
try:
    version_file_contents = version_file.read()
finally:
    version_file.close()

exec(compile(version_file_contents, "pytools/version.py", 'exec'), ver_dic)

setup(name="kernel_profiler",
      version=ver_dic["VERSION_TEXT"],
      description="A kernel profiler",
      #long_description=open("README.rst", "r").read(),

      install_requires=[
          "loo.py",
          ],

      author="James Stevens",
      #url="http://pypi.python.org/pypi/pytools",
      author_email="jdsteve2@illinois.edu",
      license="MIT",
      packages=find_packages())
