from setuptools import setup

setup(
    name="IOHMM",
    version="0.0.0",
    description='A python library for Input Output Hidden Markov Models',
    url='https://github.com/Mogeng/IOHMM',
    author='Mogeng Yin',
    author_email='mogengyin@berkeley.edu',
    license='BSD License',
    packages=['IOHMM'],
    install_requires=[
        'numpy>=1.11.0',
        'pandas>=0.19.0',
        'scikit-learn>=0.18.0',
        'scipy>=0.19.0',
        'statsmodels>=0.8.0'
    ],
    extras_require={
        'tests': [
            'flake8>=2.5.4',
            'mock>=2.0.0',
            'nose>=1.3.4',
            'coveralls>=1.1',
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
        'Programming Language :: Python :: 2.7'
    ],
)
