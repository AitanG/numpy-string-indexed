import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='numpy-string-indexed',
    version='0.0.3',
    author='Aitan Grossman',
    author_email='aitan.gros@gmail.com',
    description='A NumPy extension that allows arrays to be indexed using labels',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AitanG/numpy-string-indexed',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.6',
)