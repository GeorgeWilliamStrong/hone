from setuptools import setup, find_packages

setup(
    name="hone",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "textgrad==0.1.5",
        "instill-sdk==0.15.1",
    ],
    author="George Strong",
    author_email="george.strong@instill.tech",
    description=(
        "Automates prompt and pipeline optimization for Instill VDP "
        "(Versatile Data Pipeline) and Instill Model."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/georgewilliamstrong/hone",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9,<3.12",
)
