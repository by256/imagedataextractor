
import setuptools

with open('./README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='imagedataextractor',
    version='2.0.3',
    description='imagedataextractor is a Python library for electron microscopy image quantification.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Batuhan Yildirim',
    author_email='by256@cam.ac.uk',
    url='https://github.com/by256/imagedataextractor',
    packages=setuptools.find_packages(),
    package_data={'imagedataextractor.models': ['frozen_east_text_detection.pb', 
                                                'seg-model.pt'], 
                  'imagedataextractor.analysis.shapes': ['*.png']}, 
    include_package_data=True, 
    install_requires=['numpy>=1.19',
                      'matplotlib==2.2.4',  
                      'torch', 
                      'opencv-python>=4.2.0.32', 
                      'pytesseract==0.3.3',
                      'rdfpy>=1.0.0', 
                      'pandas', 
                      'chemdataextractor', 
                      'scikit-image'],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ])
