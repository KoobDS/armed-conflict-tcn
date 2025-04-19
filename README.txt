
# README.txt

## DESCRIPTION

Armed conflicts impose severe human and economic costs, undermining regional stability, development, and investment. We sought to predict the number of weekly armed conflict events and fatalities in an administrative region over a six-month span and to understand the impact of these events, enabling proactive government and NGO interventions. By integrating modern deep learning architectures and studying macroeconomic indicators, we aim to produce actionable insights for policymakers and humanitarian organizations to proactively allocate resources and mitigate impacts.

Previous research documents the adverse economic effects of conflict, demonstrating GDP declines and reduced human development following violence (Abadie & Gardeazabal, 2003; Novta & Pugacheva, 2021). Machine learning methods—including random forests—have improved conflict prediction but rarely integrate economic forecasting (Ahmed et al., 2023; Radford, 2022). Spatial deep learning models such as graph neural networks have shown promise for high-resolution forecasting in related domains (Chen & Lee, 2022; MirhoseiniNejad et al., 2024). However, most studies treat conflict prediction and economic impact analysis separately, limiting practical policy applications (Johnson & Martinez, 2024; Wilson & Kim, 2022). Our approach bridges this gap by modeling conflict propagation and explaining its socioeconomic consequences, offering a novel, interpretable framework for proactive decision making.

The Random Forest (RF) model performs better at predicting battles and explosions, while the Temporal Convolutional Network (TCN) performs better at predicting protests, civilian violence, riots, and fatalities. Based on these results, we selected an ensemble of both models as the final model.

The goal of the socio-economic impact analysis is to investigate the link between conflicts and economic outcomes, particularly the relationship between conflict-related fatalities and the Human Development Index (HDI).

## PREREQUISITES

Before installation, ensure you have the following:

- Stable internet connection for downloading datasets and packages.

- Administrative access to install software.

- Basic familiarity with Python, R, and running scripts.

- Recommended operating system: Windows 10+, macOS, or Linux.

- (Optional) Access to cloud computing resources (e.g., Google Colab) for deep learning model training.

## INSTALLATION

### Conflict Forecasting Files

1. **Collect ACLED Data**
   - Create an ACLED Access Portal account and generate an Access Key: https://acleddata.com/knowledge-base/acled-access-guide/
   - Use the Data Export Tool without modifying any filters, then press the "Export" button to download all data.

2. **Install Python**
   - Download Python for your operating system: https://www.python.org/downloads
   - Install your favorite IDE if needed (e.g., VS Code, PyCharm).

3. **Set Up Power BI**
   - Download and install Power BI Desktop: https://learn.microsoft.com/en-us/power-bi/fundamentals/desktop-getting-started

### Socio-Economic Impact Data Files

1. **Install R and RStudio**
   - Install R: https://www.r-project.org/
   - Install RStudio: https://posit.co/downloads/

2. **Install Necessary R Packages**
   - Install packages as needed (e.g., `WDI` package): https://www.rdocumentation.org/packages/WDI/versions/2.7.9

3. **Download External Datasets**
   - **ACLED Data**: Follow ACLED download instructions.
   - **HDI Data**: Download from Peace Research Institute Oslo (PRIO) and the Global Data Lab: https://globaldatalab.org/shdi/
   - **World Development Indicators (WDI)**: Download via WDI R package.

4. **Set Up Tableau**
   - Download Tableau: https://www.tableau.com/trial/tableau-software

## EXECUTION

### Socio-Economic Impact Scripts

1. Open `Socio-economic impact modeling.R` in RStudio.
2. Run the script to build models and generate outputs.
3. Intermediate outputs and final results will be saved as CSV files.
4. Open `EDA.twb` in Tableau to view exploratory data analysis visualizations.
5. Open `Africa Summary.twb` in Tableau to interact with the socio-economic impact dashboard.

### Conflict Forecasting Scripts

1. Place the raw ACLED dataset in the same directory as `Processing.ipynb`.
2. Run `Processing.ipynb` to clean the data. Outputs will be saved as CSV and NPY files.
3. Run `Baseline_RF.ipynb` to train Random Forest models and generate forecasts.
4. (Optional) For Deep Learning Model:
   - Use cloud resources like Google Colab for training.
   - Run `train_TCN.py` to train the Temporal Convolutional Network (TCN) model.
   - Use `curve.py` to evaluate the learning curve and check for overfitting.
5. Load the `Project_Visualization_Predicting_Armed_Conflict_Spread.pbix` file into Power BI.
   - Visualize forecasted conflict events and interpret results.

### Final Note

By following the steps above, users can replicate all analyses and visualizations from our study on armed conflict prediction and socio-economic impact analysis.
