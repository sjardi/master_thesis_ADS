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
            # 'alibaba = asreviewcontrib.models:Alibaba',
            # 'fasttext = asreviewcontrib.models:Fasttext',
            # 'bart = asreviewcontrib.models:Bart',

            # 'tfidfrelu = asreviewcontrib.models:TfidfRelu',
            'tfidfn = asreviewcontrib.models:Tfidfn',
            'tfidfsigmoid = asreviewcontrib.models:TfidfSigmoid',
            'tfidfminmax = asreviewcontrib.models:TfidfMinMax',
            # 'tfidflognorm = asreviewcontrib.models:TfidfLogNormalization',
            'tfidfaddabsmin = asreviewcontrib.models:TfidfAddAbsMin',
            'tfidfcdf = asreviewcontrib.models:Tfidfcdf',
            
            # minmax
            'tfidf_zscore_minmax = asreviewcontrib.models:Tfidf_zscore_minmax',
            'tfidf_pareto_minmax = asreviewcontrib.models:Tfidf_pareto_minmax',
            'tfidf_l2_normalize_minmax = asreviewcontrib.models:Tfidf_l2_normalize_minmax',
            # absmin
            'tfidf_zscore_absmin = asreviewcontrib.models:Tfidf_zscore_absmin',
            'tfidf_pareto_absmin = asreviewcontrib.models:Tfidf_pareto_absmin',
            'tfidf_l2_normalize_absmin = asreviewcontrib.models:Tfidf_l2_normalize_absmin',
            # sqrt
            'tfidf_zscore_sqrt = asreviewcontrib.models:Tfidf_zscore_sqrt',
            'tfidf_pareto_sqrt = asreviewcontrib.models:Tfidf_pareto_sqrt',
            'tfidf_l2_normalize_sqrt = asreviewcontrib.models:Tfidf_l2_normalize_sqrt',
            # # cdf
            'tfidf_zscore_cdf = asreviewcontrib.models:Tfidf_zscore_cdf',
            'tfidf_pareto_cdf = asreviewcontrib.models:Tfidf_pareto_cdf',
            'tfidf_l2_normalize_cdf = asreviewcontrib.models:Tfidf_l2_normalize_cdf',
            # sigmoid
            'tfidf_zscore_sigmoid = asreviewcontrib.models:Tfidf_zscore_sigmoid',
            'tfidf_pareto_sigmoid = asreviewcontrib.models:Tfidf_pareto_sigmoid',
            'tfidf_l2_normalize_sigmoid = asreviewcontrib.models:Tfidf_l2_normalize_sigmoid',
            ## Doc2Vec
            # minmax
            'doc2vec_zscore_minmax = asreviewcontrib.models:Doc2Vec_zscore_minmax',
            'doc2vec_pareto_minmax = asreviewcontrib.models:Doc2Vec_pareto_minmax',
            'doc2vec_minmax_l2_normalize = asreviewcontrib.models:Doc2Vec_minmax_l2_normalize',
            # absmin
            'doc2vec_zscore_absmin = asreviewcontrib.models:Doc2Vec_zscore_absmin',
            'doc2vec_pareto_absmin = asreviewcontrib.models:Doc2Vec_pareto_absmin',
            'doc2vec_l2_normalize_absmin = asreviewcontrib.models:Doc2Vec_l2_normalize_absmin',
            # sqrt
            'doc2vec_zscore_sqrt = asreviewcontrib.models:Doc2Vec_zscore_sqrt',
            'doc2vec_pareto_sqrt = asreviewcontrib.models:Doc2Vec_pareto_sqrt',
            'doc2vec_l2_normalize_sqrt = asreviewcontrib.models:Doc2Vec_l2_normalize_sqrt',
            # cdf
            'doc2vec_zscore_cdf = asreviewcontrib.models:Doc2Vec_zscore_cdf',
            'doc2vec_pareto_cdf = asreviewcontrib.models:Doc2Vec_pareto_cdf',
            'doc2vec_l2_normalize_cdf = asreviewcontrib.models:Doc2Vec_l2_normalize_cdf',
            # sigmoid
            'doc2vec_zscore_sigmoid = asreviewcontrib.models:Doc2Vec_zscore_sigmoid',
            'doc2vec_pareto_sigmoid = asreviewcontrib.models:Doc2Vec_pareto_sigmoid',
            'doc2vec_l2_normalize_sigmoid = asreviewcontrib.models:Doc2Vec_l2_normalize_sigmoid',
            #
            
            #SBERT
            # MinMax
            'sbert_zscore_minmax = asreviewcontrib.models:SBERT_zscore_minmax',
            'sbert_pareto_minmax = asreviewcontrib.models:SBERT_pareto_minmax',
            'sbert_l2_normalize_minmax = asreviewcontrib.models:SBERT_l2_normalize_minmax',
            #
            'sbert_zscore_absmin = asreviewcontrib.models:SBERT_zscore_absmin',
            'sbert_pareto_absmin = asreviewcontrib.models:SBERT_pareto_absmin',
            'sbert_l2_normalize_absmin = asreviewcontrib.models:SBERT_l2_normalize_absmin',
            #
            'sbert_zscore_sqrt = asreviewcontrib.models:SBERT_zscore_sqrt',
            'sbert_pareto_sqrt = asreviewcontrib.models:SBERT_pareto_sqrt',
            'sbert_l2_normalize_sqrt = asreviewcontrib.models:SBERT_l2_normalize_sqrt',
            #
            'sbert_zscore_cdf = asreviewcontrib.models:SBERT_zscore_cdf',
            'sbert_pareto_cdf = asreviewcontrib.models:SBERT_pareto_cdf',
            'sbert_l2_normalize_cdf = asreviewcontrib.models:SBERT_l2_normalize_cdf',
            #
            'sbert_zscore_sigmoid = asreviewcontrib.models:SBERT_zscore_sigmoid',
            'sbert_pareto_sigmoid = asreviewcontrib.models:SBERT_pareto_sigmoid',
            'sbert_l2_normalize_sigmoid = asreviewcontrib.models:SBERT_l2_normalize_sigmoid',
            #
            
            
            
            'doc2vec = asreviewcontrib.models:Doc2Vec',
            'doc2vecaddabsmin = asreviewcontrib.models:Doc2VecAddAbsMin',
            'doc2vecminmax = asreviewcontrib.models:Doc2VecMinMax',
            # 'doc2vecsoftplus = asreviewcontrib.models:Doc2VecSoftplus',
            'doc2veccdf = asreviewcontrib.models:Doc2Veccdf',
            'doc2vecsigmoid = asreviewcontrib.models:Doc2VecSigmoid',

            'sbertminmax = asreviewcontrib.models:SBERTMinMax',
            'sbertabsmin = asreviewcontrib.models:SBERTabsmin',
            'sbertsigmoid = asreviewcontrib.models:SBERTSigmoid',
            'sbert_cdf = asreviewcontrib.models:SBERT_cdf',
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
