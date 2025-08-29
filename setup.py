from setuptools import find_packages, setup

setup(
    name="three_d_medical_segmentation",
    version="0.1.0",
    description="Comparative analysis framework for 3D medical image segmentation",
    author="Thabhelo Duve",
    packages=find_packages(exclude=("tests", "notebooks", "results")),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "monai[all]>=1.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "nibabel>=5.1.0",
        "SimpleITK>=2.3.0",
        "PyYAML>=6.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.14.0",
        "jupyter>=1.0.0"
    ],
)
