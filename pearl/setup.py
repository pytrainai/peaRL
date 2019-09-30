from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pearl',
      version='0.2.1',
      description='Python Experiments for Agents in RL',
      url='https://github.com/pytrainai/pearl',
      author='PyTrain.ai',
      author_email='pytrainteam@gmail.com',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=setuptools.find_packages(),
      license='MIT',
      zip_safe=False)