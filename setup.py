from setuptools import find_packages, setup

setup(
    name="three_d_medical_segmentation",
    version="0.1.0",
    description="Comparative analysis framework for 3D medical image segmentation",
    author="Thabhelo Duve",
    packages=find_packages(exclude=("tests", "notebooks", "results")),
    python_requires=">=3.9",
    install_requires=[
        # Torch/TV left unpinned here; prefer environment-specific wheels (e.g., cu121) in notebooks/CI
        "torch>=2.3",
        "torchvision>=0.18",
        # Avoid monai[all] to reduce optional extras; >=1.4 is Python 3.12 friendly
        "monai>=1.4",
        "numpy>=1.26",
        "scipy>=1.12",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "nibabel>=5.1",
        "SimpleITK>=2.3",
        "PyYAML>=6.0",
        "tqdm>=4.66",
        "tensorboard>=2.14",
        "jupyter>=1.0",
    ],
)
