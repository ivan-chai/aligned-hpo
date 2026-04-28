import os
import setuptools

with open("README.md") as fp:
    long_description = fp.read()


setuptools.setup(
    name="aligned-hpo",
    version="0.0.1",
    author="Ivan Karpukhin",
    author_email="karpuhini@yandex.ru",
    description="SGD-based optimizer for differentiable hyperparameters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["aligned_hpo", "aligned_hpo.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "cvxopt>=1.3.0",
        "datasketches>=5.0.0",
        "numpy>=1.23",
        "qpsolvers>=4.0.0",
        "scipy>=1.11",
        "torch>=1.12.0"
    ]
)
