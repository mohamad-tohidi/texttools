from setuptools import setup, find_packages

setup(
    name="texttools",
    version="0.0.7",
    packages=find_packages(),
    author="Tohidi",
    description="set of my text tools",
    install_requires=[
        "openai==1.77.0",
        "jsonschema==4.23.0"
    ]    
)
