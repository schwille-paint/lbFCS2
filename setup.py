from setuptools import setup

setup(
   name='lbfcs2',
   version='1.0.0',
   description='Calibration free molecular counting and hyridization rates in single DNA-PAINT localization clusters',
   license="MIT License",
   author='Stehr Florian',
   author_email='florian.stehr@gmail.com',
   url="http://www.github.com/schwille-paint/lbFCS2",
   packages=['lbfcs'],
   classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
   ],
   install_requires=[
		'picasso @ git+https://github.com/jungmannlab/picasso.git',
        'picasso_addon @ git+https://github.com/schwille-paint/picasso_addon.git',
   ], 
)
