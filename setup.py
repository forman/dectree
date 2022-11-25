from setuptools import setup

setup(
    name='dectree',
    version='0.0.1',
    packages=['test', 'dectree'],
    url='',
    license='MIT',
    author='Norman Fomferra',
    author_email='',
    description='Fuzzy Decision Tree',
    requires=['numba', 'numpy', 'pyyaml'],
    entry_points={
        'console_scripts': [
            'dectree = dectree.main:main',
        ],
    },
)
