from setuptools import setup, find_packages


classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'Programming Language :: Python :: 3.8',
    'License :: OSI Approved :: MIT License'
]

setup(
    name='mloptima',
    version='0.0.1',
    description='Optimization methods, machine learning tools and models with visualization',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/M1croZavr/ML_opt',
    author='Timofey Lashukov',
    author_email='tlashukov@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='ml visualization optimization',
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn', 'matplotlib', 'sympy', 'pandas', 'scipy']
)