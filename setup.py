from setuptools import setup, find_packages

setup(
  name = 'antiberty-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.0.1',
  license='MIT',
  description = 'Antiberty - Pytorch',
  author = 'Dohoon Lee',
  author_email = 'dohlee.bioinfo@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/dohlee/antiberty-pytorch',
  keywords = [
    'artificial intelligence',
    'antibody',
    'protein language model',
  ],
  install_requires=[
    'einops>=0.3',
    'numpy',
    'torch>=1.6',
    'transformers',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)
