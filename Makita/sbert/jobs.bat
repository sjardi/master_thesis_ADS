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


:: Skipped nb + sbert + max model

:: Classifier = nb, Feature extractor = sbertminmax, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertminmax_max_double.asreview --model nb --query_strategy max --feature_extraction sbertminmax --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertminmax_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_sbertminmax_max_double.json

:: Classifier = nb, Feature extractor = sbertabsmin, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertabsmin_max_double.asreview --model nb --query_strategy max --feature_extraction sbertabsmin --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertabsmin_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_sbertabsmin_max_double.json

:: Classifier = nb, Feature extractor = sbertsigmoid, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertsigmoid_max_double.asreview --model nb --query_strategy max --feature_extraction sbertsigmoid --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertsigmoid_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_sbertsigmoid_max_double.json

:: Classifier = nb, Feature extractor = sbertcdf, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertcdf_max_double.asreview --model nb --query_strategy max --feature_extraction sbertcdf --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_nb_sbertcdf_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_nb_sbertcdf_max_double.json

:: Classifier = logistic, Feature extractor = sbert, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbert_max_double.asreview --model logistic --query_strategy max --feature_extraction sbert --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbert_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_sbert_max_double.json

:: Classifier = logistic, Feature extractor = sbertminmax, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertminmax_max_double.asreview --model logistic --query_strategy max --feature_extraction sbertminmax --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertminmax_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_sbertminmax_max_double.json

:: Classifier = logistic, Feature extractor = sbertabsmin, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertabsmin_max_double.asreview --model logistic --query_strategy max --feature_extraction sbertabsmin --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertabsmin_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_sbertabsmin_max_double.json

:: Classifier = logistic, Feature extractor = sbertsigmoid, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertsigmoid_max_double.asreview --model logistic --query_strategy max --feature_extraction sbertsigmoid --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertsigmoid_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_sbertsigmoid_max_double.json

:: Classifier = logistic, Feature extractor = sbertcdf, Query strategy = max, Balance strategy = double
python -m asreview simulate data\van_de_Schoot_2018.csv -s output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertcdf_max_double.asreview --model logistic --query_strategy max --feature_extraction sbertcdf --init_seed 535 --seed 165 -q max -b double --n_instances 1 --stop_if min
python -m asreview metrics output\simulation\van_de_Schoot_2018\state_files\sim_van_de_Schoot_2018_logistic_sbertcdf_max_double.asreview -o output\simulation\van_de_Schoot_2018\metrics\metrics_sim_van_de_Schoot_2018_logistic_sbertcdf_max_double.json

:: Generate plot and tables for dataset
python scripts\get_plot.py -s output\simulation\van_de_Schoot_2018\state_files\ -o output\figures\plot_recall_sim_van_de_Schoot_2018.png --show_legend model
python scripts\merge_metrics.py -s output\simulation\van_de_Schoot_2018\metrics\ -o output\tables\metrics\metrics_sim_van_de_Schoot_2018.csv
python scripts\merge_tds.py -s output\simulation\van_de_Schoot_2018\metrics\ -o output\tables\time_to_discovery\tds_sim_van_de_Schoot_2018.csv

:: Merge descriptives and metrics
python scripts\merge_descriptives.py
python scripts\merge_metrics.py
