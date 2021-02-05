from setuptools import setup

setup(
    name="IOHMM",
    version="0.0.3",
    description='A python library for Input Output Hidden Markov Models',
    url='https://github.com/Mogeng/IOHMM',
    author='Mogeng Yin',
    author_email='mogengyin@berkeley.edu',
    license='BSD License',
    packages=['IOHMM'],
    install_requires=[
        'numpy >= 1.20.0',
        'future >= 0.18.2',
        'pandas >= 1.2.1',
        'scikit-learn >= 0.24.1',
        'scipy >= 1.6.0',
        'statsmodels >= 0.12.2',
    ],
    extras_require={
        'tests': [
            'flake8>=3.8.4',
            'mock>=3.9.1',
            'nose>=1.3.7',
            'coveralls>=3.0.0',
            'pytest',
        ]
    },
    zip_safe=True,
    keywords=' '.join([
        'python',
        'hidden-markov-model',
        'graphical-models',
        'sequence-to-sequence',
        'machine-learning',
        'linear-models',
        'sequence-labeling',
        'supervised-learning',
        'semi-supervised-learning',
        'unsupervised-learning',
        'time-series',
        'scikit-learn',
        'statsmodels']),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)
