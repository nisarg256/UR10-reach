from setuptools import setup, find_packages

setup(
    name="ur10_reach_analysis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "mujoco==2.3.2",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "pyrender>=0.1.45",
        "trimesh>=3.9.0",
    ],
    python_requires=">=3.8",
) 