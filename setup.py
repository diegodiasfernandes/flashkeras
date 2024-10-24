from setuptools import setup, find_packages # type: ignore

setup(
    name="flashkeras",
    version="0.1.5",
    packages=find_packages(include=['flashkeras', 'flashkeras.*']),
    install_requires=[ 
        'tensorflow==2.15.0',
        'matplotlib>=3.9.2',
        'opencv-python>=4.10.0.84',
        'pandas>=2.2.2',
        'scikit-learn>=1.5.1'
    ],
    author="diego dias fernandes",
    author_email="diegodiasfernandes.comp@gmail.com",
    description="Package to fasten Keras manipulation and overall machine learning pipeline.",
    url="https://github.com/diegodiasfernandes/flashkeras",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10,<3.12',
)
