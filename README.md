# Armed Conflict Analysis:

This is a study of armed conflict, including analysis of impacts on human and economic development using classical methods and projecting future incidence using deep learning.
We use Armed Conflict Location & Event Data (ACLED) for this study, including their integrated population estimates.
Raleigh, C., Kishi, R. & Linke, A. Political instability patterns are obscured by conflict dataset scope conditions, sources, and coding choices. Humanit Soc Sci Commun 10, 74 (2023). https://doi.org/10.1057/s41599-023-01559-4
https://acleddata.com/data/

# Methodology

Our goal was to predict the number of weekly violent-conflicts and fatalities per "actor", where actors are akin to states / providences. Similar to many of the literature we found, we intializied a baseline Random Forest model to predict the compare our results against. Since Random Forest doesn't natively capture temporality, we used lags of the different event counts and the population. To keep the comparison fair, we did not do any recursive predictions or forecasting, we were just trying to predict one week ahead. After training the baseline Random Forest, we predicted the next 6 months of data, only using lags.

Unlike most of the literature relating to political events, we decided to use a Temporal Convolutional Network (TCN) to predict 26-week violent-conflict dynamics (6 targets) from 52-weeks of history. This modern deep learning architecture is designed to retain temporal effects and has been shown to work well in other fields. Due to time constraints and lack of computer power, we tried to make the architecture as simple as possible while still being effective. The model reads hyper-parameters from a YAML config, which helps with source control. The model uses mixed-precision (AMP) + cuDNN autotuner for speed. Lastly, it saves metrics and flat per-week predictions to output.directory.

# Results

The Random Forest baseline performed surprsingly good:

| Target Variable | MAE | MSE | R^2 |
| --- | --- | --- | --- |
| fatalities | 1.1209 | 16.3033 | 0.3483 |
| count_battles | 0.1724 | 1.422 | 0.8591 |
| count_protests | 0.6159 | 2.1798 | 0.6373 |
| count_riots	0.1147 | 0.6513 |	0.8019 |
| count_explosions | 0.1685 |	1.9576 | 0.9295 |
| count_civ_violence | 0.1595 | 0.623 | 0.6645 |

Fatalities can vary a lot and is not strongly related to the type of conflict event. Thus, due to its volatility, it makes sense why the model struggled to predict it. 

# Discussion

# Conclusion
