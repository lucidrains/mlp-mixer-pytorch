from setuptools import setup, find_packages

setup(
  name = 'mlp-mixer-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.2.0',
  license='MIT',
  description = 'MLP Mixer - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/mlp-mixer-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'image recognition'
  ],
  install_requires=[
    'einops>=0.8',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
