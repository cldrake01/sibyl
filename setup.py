from setuptools import setup, find_packages

setup(
    name="sibyl",
    version="0.1",
    author="cldrake01",
    author_email="collinlindendrake@gmail.com",
    description="A package for stock price prediction using deep learning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cldrake01/divine-inference",
    packages=find_packages(),
    python_requires=">=3.10",
    include_package_data=True,
)
