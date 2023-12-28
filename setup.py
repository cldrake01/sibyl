from setuptools import setup, find_packages

setup(
    name="divine_inference",
    url="https://github.com/cldrake01/divine-inference",
    author="cldrake01",
    author_email="collinlindendrake@gmail.com",
    version="0.1",
    packages=find_packages(),
    description="A package for stock price prediction using deep learning.",
    long_description=open("README.md").read(),
)
