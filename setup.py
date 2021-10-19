from setuptools import setup

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()
    
    
setup(
    name="src",
    version="0.0.1",
    author="raja67",
    description="A small package for ANN Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NanbanRaj/ANN-Implementation.git",
    author_email="rajasimman67@gmail.com",
    package =["src"],
    python_requires=">=3.7",
    install_requires=[
        'tensorflow',
        'matplotlib',
        'seaborn',
        'numpy',
        'pandas'

    ]
)
    