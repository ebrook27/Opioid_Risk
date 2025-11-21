# Opioid Risk Quantification

This repo contains modelling, data processing, and visualization pipelines for identifying US counties at high risk of opioid overdose mortality.
We extend empirical minimization to identify the counties persistently at high risk, and then perform case studies on several particularly high risk counties.  
Uses uv to handle the virtual environment, and supports argument parsing. As we add models we'll have to expand the input args.  
To run from the terminal, navigate to this repo and run:  
```
uv run python main.py --model xgboost --plot risk
```
The model and plot can be changed, and the save directory of the plots can be specified with "--save_dir".

# TODOs:
- [X] Run risk pipeline for several different model types.
- [X] Perform case-study analysis for couple counties.
- [ ] Set up Gain, Frequency, Cover feature importance analysis framework (see Huang and Huang, doi.org/10.29024/jsim.181)
- [X] Add map plotting function with MR, Equal-Weight Risk, and EWMA risk.