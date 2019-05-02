from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='cav-keras',
    version='0.1.0',
    description='Package for concept activation vectors (CAVs) in Keras',
    long_description=readme,
    author='Peter Xenopoulos',
    author_email='xenopoulos@nyu.edu',
    url='https://github.com/pnxenopoulos/cav-keras',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
