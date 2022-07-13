from setuptools import setup, find_packages

setup(
    name='baseline',
    version='0.1.0',
    description='quick baseline models using sklearn',
    license="MIT",
    author='Vito Stamatti',
    package_dir={'':'.'},
    packages=find_packages(where='.'),
    install_requires=[
        'scikit-learn', 
    ],
),