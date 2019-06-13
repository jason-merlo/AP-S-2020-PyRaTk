import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyratk",
    version="0.0.1-dev1",
    author="Jason Merlo",
    author_email="merlojas@msu.edu",
    description="Collection of tools for radar data acquisition, processing, and visualization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.msu.edu/delta/python_radar_toolkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
