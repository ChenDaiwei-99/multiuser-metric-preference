# One for All: Simultaneous Metric and Preference Learning over Multiple Users

Paper code for "[One for All: Simultaneous Metric and Preference Learning over Multiple Users](https://arxiv.org/pdf/2207.03609.pdf)"

## Requirements
- Python 3.9.7
- cvxpy 1.2.0
- matplotlib 3.4.3
- numpy 1.21.2
- pyyaml 6.0
- scikit-learn 1.0
- scipy 1.7.1

## Code structure
- `cluster_script.py`: main script to run pooled experiments from configuration file

    Example usage: `python cluster_script.py --config ./config.yaml`

- `run_experiments.py`: script to run isolated experiments (called by `cluster_script.py`)
- `dataset.py`: defines dataset object
- `multipref.py`: defines MultiPref model object
- `utils.py`: defines utility functions
- `metrics.py`: defines performance metrics
- `analysis_utils.py`: utilities for data analysis and plotting results. Generates paper plots when run as standalone file (note, paths are set for author's computer)
- `heatmap.py`: script for generating metric heatmap
- `./configs`: configuration files used to run experiments

## Color data structure
The color preference data is contained in `CPdata.mat`. The variables are as follows:
- `CIELAB`: 37 x 3 double of color coordinates in CIELAB space, where each row is a separate color and each column is a CIELAB coordinate
- `SingPrefAFCload`: 37 x 37 x 48 double structured as follows. The third axis corresponds to each respondent, i.e., each 37 x 37 slice along the third axis defines the preference data for each individual user. Each slice is a grid of binary preferences over all possible color pairs among the 37 colors in the dataset. During data collection, a color pair was presented to each respondent on a screen, with one color on the left of the screen and the other color on the right. The rows of each participant's 37 x 37 slice index the colors presented on the left, and the columns index the colors presented on the right. A value of 1 indicates that the left color was preferred, while a value of 2 indicates that the color on the right was preferred. The diagonal is filled with zeros, since pairs with identical colors were not queried.

Data is provided by courtesy of Professor Karen Schloss (University of Wisconsin-Madison) and was collected from the same participants in *Palmer, S. E., & Schloss, K. B. (2010). An ecological valence theory of human color preference. Proceedings of the National Academy of Sciences, 107(19), 8877-8882.*
