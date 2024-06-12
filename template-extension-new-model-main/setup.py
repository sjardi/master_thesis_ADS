from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='asreview-template-model-extension',
    version='1.0',
    description='Example classifier extension',
    url='https://github.com/asreview/asreview',
    author='ASReview team',
    author_email='asreview@uu.nl',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='systematic review',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    python_requires='~=3.6',
    install_requires=[
        'asreview>=1.0'
    ],
    entry_points={
        'asreview.models.classifiers': [
            #
        ],
        'asreview.models.feature_extraction': [
            # define feature_extraction algorithms
            'alibaba = asreviewcontrib.models:Alibaba',
            'fasttext = asreviewcontrib.models:Fasttext',
            'bart = asreviewcontrib.models:Bart',
            
            'tfidfrelu = asreviewcontrib.models:TfidfRelu',
            'tfidfn = asreviewcontrib.models:Tfidfn',
            'tfidfsigmoid = asreviewcontrib.models:TfidfSigmoid',
            'tfidfminmax = asreviewcontrib.models:TfidfMinMax',
            'tfidflognorm = asreviewcontrib.models:TfidfLogNormalization',
            'tfidfaddabsmin = asreviewcontrib.models:TfidfAddAbsMin',
            'tfidfcdf = asreviewcontrib.models:Tfidfcdf',
            
            'doc2vec = asreviewcontrib.models:Doc2Vec',
            'doc2vecaddabsmin = asreviewcontrib.models:Doc2VecAddAbsMin',
            'doc2vecminmax = asreviewcontrib.models:Doc2VecMinMax',
            'doc2vecsoftplus = asreviewcontrib.models:Doc2VecSoftplus',
            'doc2veccdf = asreviewcontrib.models:Doc2Veccdf',
            'doc2vecsigmoid = asreviewcontrib.models:Doc2VecSigmoid',
            
            'sbertminmax = asreviewcontrib.models:SBERTMinMax',
            'sbertabsmin = asreviewcontrib.models:SBERTabsmin',
            'sbertsigmoid = asreviewcontrib.models:SBERTSigmoid',
            'sbertcdf = asreviewcontrib.models:SBERTcdf',
        ],
        'asreview.models.balance': [
            # define balance strategy algorithms
        ],
        'asreview.models.query': [
            # define query strategy algorithms
        ]
    },
    project_urls={
        'Bug Reports': 'https://github.com/asreview/asreview/issues',
        'Source': 'https://github.com/asreview/asreview/',
    },
)
