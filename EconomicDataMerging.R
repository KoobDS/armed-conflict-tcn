# Load required libraries
library(tidyverse)   
library(readr)      
library(haven)       

# Set working directory
setwd("D:/Konstantinos Vrachimis/Georgia Tech/CSE 6242/Group Work/")

# Import HDI Data
# Source: https://globaldatalab.org/shdi/table/
hdi_data <- read_csv("SHDI-SGDI-Total 8.0.csv")

# Convert all column names to lowercase for consistency
hdi_data <- hdi_data %>%
  rename_with(tolower)

# Convert specified columns from character to numeric
numeric_cols <- c("sgdi", "shdi", "shdif", "shdim", 
                  "healthindex", "healthindexf", "healthindexm",
                  "incindex", "incindexf", "incindexm",
                  "edindex", "edindexf", "edindexm",
                  "esch", "eschf", "eschm",
                  "msch", "mschf", "mschm",
                  "lifexp", "lifexpf", "lifexpm",
                  "gnic", "gnicf", "gnicm",
                  "lgnic", "lgnicf", "lgnicm", "pop")

hdi_data <- hdi_data %>%
  mutate(across(all_of(numeric_cols), as.numeric))

# Save HDI dataset as a .dta file
write_dta(hdi_data, "HDI_Dataset.dta")

# -------------------------------------------------------------------
# Import World Bank Data (WDI)
# -------------------------------------------------------------------

# Load WDI data (CSV version)
# Source:https://datacatalog.worldbank.org/search/dataset/0037712/World-Development-Indicators"
setwd("D:/Konstantinos Vrachimis/Georgia Tech/CSE 6242/Group Work/WDI_CSV_2025_01_28")
wdi_data <- read_csv("WDICSV.csv")

# Convert all column names to lowercase for consistency
wdi_data <- wdi_data %>%
  rename_with(tolower)

# Print column names for debugging
print(colnames(wdi_data))

# Identify the correct column name for Indicator Code
possible_cols <- colnames(wdi_data)[grepl("indicator", colnames(wdi_data), ignore.case = TRUE)]
print(possible_cols)  # Check matched columns

# Select the correct column name (prefer "indicator code" over "indicator name")
indicator_col <- possible_cols[grepl("code", possible_cols, ignore.case = TRUE)][1]

# If "indicator code" is not found, default to "indicator name"
if (is.na(indicator_col)) {
  indicator_col <- possible_cols[1]  # Use the first matched column
}

# Identify the correct column for country codes
country_col <- colnames(wdi_data)[grepl("country.*code", colnames(wdi_data), ignore.case = TRUE)][1]

# Print the detected country code column
print(paste("Using country code column:", country_col))

# Identify columns that contain years (e.g., "1960", "2000", etc.)
year_cols <- colnames(wdi_data)[grepl("\\d{4}", colnames(wdi_data))]

# Print detected year columns
print(year_cols)

# Ensure that at least one year column exists
if (length(year_cols) == 0) {
  stop("No year columns detected. Please check the column names in WDICSV.csv.")
}

# Filter for the indicator of interest: "IS.RRS.TOTL.KM"
wdi_filtered <- wdi_data %>%
  filter(.data[[indicator_col]] == "IS.RRS.TOTL.KM") %>%  # Use dynamically found column
  pivot_longer(cols = all_of(year_cols),  # Use detected year columns
               names_to = "year", 
               values_to = "rail_line") %>%
  mutate(year = as.numeric(year),  # Ensure year is numeric
         iso_code = .data[[country_col]]) %>%  # Use detected country code column
  select(iso_code, year, rail_line)  # Keep relevant columns

# Save ports data
write_dta(wdi_filtered, "ports_data.dta")

# -------------------------------------------------------------------
# Merge HDI and WDI Data
# -------------------------------------------------------------------

# Load HDI dataset
setwd("D:/Konstantinos Vrachimis/Georgia Tech/CSE 6242/Group Work/")
hdi_data <- read_dta("HDI_Dataset.dta")

# Merge with WDI data
merged_data <- hdi_data %>%
  full_join(wdi_filtered, by = c("year", "iso_code"))

# Save merged dataset
write_dta(merged_data, "Merged_Dataset.dta")

