"""Setup script for HeadHunt-VAD."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="headhunt-vad",
    version="0.1.0",
    author="HeadHunt-VAD Authors",
    description="HeadHunt-VAD: Tuning-Free Video Anomaly Detection via Robust Head Identification in MLLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/headhunt-vad/headhunt-vad",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "headhunt-extract=scripts.extract_features:main",
            "headhunt-rhi=scripts.run_rhi:main",
            "headhunt-train=scripts.train_scorer:main",
            "headhunt-infer=scripts.inference:main",
            "headhunt-eval=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
)
