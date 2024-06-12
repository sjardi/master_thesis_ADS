@ echo off
COLOR E0
:: version 0.0.0

:: Create folder structure. By default, the folder 'output' is used to store output.
mkdir output
mkdir output\simulation
mkdir output\tables
mkdir output\tables\metrics
mkdir output\tables\time_to_discovery
mkdir output\figures

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::: DATASET: van_de_Schoot_2018
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:: Create output folder
mkdir output\simulation\van_de_Schoot_2018\
mkdir output\simulation\van_de_Schoot_2018\metrics

:: Collect descriptives about the dataset van_de_Schoot_2018
mkdir output\simulation\van_de_Schoot_2018\descriptives
python -m asreview data describe data\van_de_Schoot_2018.csv -o output\simulation\van_de_Schoot_2018\descriptives\data_stats_van_de_Schoot_2018.json

:: Generate wordcloud visualizations of all datasets
python -m asreview wordcloud data\van_de_Schoot_2018.csv -o output\figures\wordcloud_van_de_Schoot_2018.png --width 800 --height 500
python -m asreview wordcloud data\van_de_Schoot_2018.csv -o output\figures\wordcloud_relevant_van_de_Schoot_2018.png --width 800 --height 500 --relevant
python -m asreview wordcloud data\van_de_Schoot_2018.csv -o output\figures\wordcloud_irrelevant_van_de_Schoot_2018.png --width 800 --height 500 --irrelevant

:: Simulate runs
mkdir output\simulation\van_de_Schoot_2018\state_files

:: Classifier = logistic, Feature extractor = doc2vec, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_doc2vec_max_double.asreview --model logistic --query_strategy max --feature_extraction doc2vec --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_doc2vec_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_doc2vec_max_double.json

:: Classifier = logistic, Feature extractor = doc2vecaddabsmin, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_doc2vecaddabsmin_max_double.asreview --model logistic --query_strategy max --feature_extraction doc2vecaddabsmin --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_doc2vecaddabsmin_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_doc2vecaddabsmin_max_double.json

:: Classifier = logistic, Feature extractor = doc2vecminmax, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_doc2vecminmax_max_double.asreview --model logistic --query_strategy max --feature_extraction doc2vecminmax --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_doc2vecminmax_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_doc2vecminmax_max_double.json

:: Classifier = logistic, Feature extractor = doc2vecsoftplus, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_doc2vecsoftplus_max_double.asreview --model logistic --query_strategy max --feature_extraction doc2vecsoftplus --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_doc2vecsoftplus_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_doc2vecsoftplus_max_double.json

:: Classifier = logistic, Feature extractor = tfidfrelu, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_tfidfrelu_max_double.asreview --model logistic --query_strategy max --feature_extraction tfidfrelu --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_tfidfrelu_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_tfidfrelu_max_double.json

:: Classifier = logistic, Feature extractor = tfidfn, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_tfidfn_max_double.asreview --model logistic --query_strategy max --feature_extraction tfidfn --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_tfidfn_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_tfidfn_max_double.json

:: Classifier = logistic, Feature extractor = tfidfminmax, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_tfidfminmax_max_double.asreview --model logistic --query_strategy max --feature_extraction tfidfminmax --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_tfidfminmax_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_tfidfminmax_max_double.json

:: Classifier = logistic, Feature extractor = tfidflognorm, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_tfidflognorm_max_double.asreview --model logistic --query_strategy max --feature_extraction tfidflognorm --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_tfidflognorm_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_tfidflognorm_max_double.json

:: Classifier = logistic, Feature extractor = tfidfaddabsmin, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_tfidfaddabsmin_max_double.asreview --model logistic --query_strategy max --feature_extraction tfidfaddabsmin --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_tfidfaddabsmin_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_tfidfaddabsmin_max_double.json

:: Classifier = logistic, Feature extractor = sbertminmax, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertminmax_max_double.asreview --model logistic --query_strategy max --feature_extraction sbertminmax --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertminmax_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_sbertminmax_max_double.json

:: Classifier = logistic, Feature extractor = sbertabsmin, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertabsmin_max_double.asreview --model logistic --query_strategy max --feature_extraction sbertabsmin --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertabsmin_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_sbertabsmin_max_double.json


:: Skipped nb + doc2vec + max model

:: Classifier = nb, Feature extractor = doc2vecaddabsmin, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_doc2vecaddabsmin_max_double.asreview --model nb --query_strategy max --feature_extraction doc2vecaddabsmin --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_doc2vecaddabsmin_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_doc2vecaddabsmin_max_double.json

:: Classifier = nb, Feature extractor = doc2vecminmax, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_doc2vecminmax_max_double.asreview --model nb --query_strategy max --feature_extraction doc2vecminmax --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_doc2vecminmax_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_doc2vecminmax_max_double.json

:: Classifier = nb, Feature extractor = doc2vecsoftplus, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_doc2vecsoftplus_max_double.asreview --model nb --query_strategy max --feature_extraction doc2vecsoftplus --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_doc2vecsoftplus_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_doc2vecsoftplus_max_double.json

:: Classifier = nb, Feature extractor = tfidfrelu, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_tfidfrelu_max_double.asreview --model nb --query_strategy max --feature_extraction tfidfrelu --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_tfidfrelu_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_tfidfrelu_max_double.json

:: Classifier = nb, Feature extractor = tfidfn, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_tfidfn_max_double.asreview --model nb --query_strategy max --feature_extraction tfidfn --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_tfidfn_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_tfidfn_max_double.json

:: Classifier = nb, Feature extractor = tfidfminmax, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_tfidfminmax_max_double.asreview --model nb --query_strategy max --feature_extraction tfidfminmax --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_tfidfminmax_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_tfidfminmax_max_double.json

:: Classifier = nb, Feature extractor = tfidflognorm, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_tfidflognorm_max_double.asreview --model nb --query_strategy max --feature_extraction tfidflognorm --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_tfidflognorm_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_tfidflognorm_max_double.json

:: Classifier = nb, Feature extractor = tfidfaddabsmin, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_tfidfaddabsmin_max_double.asreview --model nb --query_strategy max --feature_extraction tfidfaddabsmin --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_tfidfaddabsmin_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_tfidfaddabsmin_max_double.json

:: Classifier = nb, Feature extractor = sbertminmax, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertminmax_max_double.asreview --model nb --query_strategy max --feature_extraction sbertminmax --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertminmax_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_sbertminmax_max_double.json

:: Classifier = nb, Feature extractor = sbertabsmin, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertabsmin_max_double.asreview --model nb --query_strategy max --feature_extraction sbertabsmin --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertabsmin_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_sbertabsmin_max_double.json

:: Generate plot and tables for dataset
python scripts\get_plot.py -s output\simulation\van_de_Schoot_2018\state_files\ -o output\figures\plot_recall_sim_van_de_Schoot_2018.png --show_legend model
python scripts\merge_metrics.py -s output\simulation\van_de_Schoot_2018\metrics\ -o output\tables\metrics\metrics_sim_van_de_Schoot_2018.csv
python scripts\merge_tds.py -s output\simulation\van_de_Schoot_2018\metrics\ -o output\tables\time_to_discovery\tds_sim_van_de_Schoot_2018.csv

:: Merge descriptives and metrics
python scripts\merge_descriptives.py
python scripts\merge_metrics.py
