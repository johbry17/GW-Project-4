# Project-4 - Insights into Housing Prices: Neural Network vs. Random Forest Analysis

Development on this project has stopped.

Refer to `predictive_models_report.md` for insight and analysis.

## Table of Contents

- [Description](#description)
- [Usage](#usage)
- [Gallery](#gallery-of-results)
- [References](#references)
- [Licenses](#licenses)
- [Acknowledgements](#acknowledgements)
- [Authors](#authors)

## Description

This project employs the classic Ames, Iowa housing dataset, spanning residential property sales from 2006 to 2010, to predict individual residential property prices using two distinct machine learning models. The experiment involves improving a neural network through feature engineering and hyperparameter tuning, alongside the construction of a random forest model for comparative analysis. The subsequent discussion in `predictive_models_report.md` delves into the strengths and weaknesses of each model, offering insights into their unique attributes and trade-offs.

### Overview

The project workflow encompassed Exploratory Data Analysis (EDA), the implementation of a Neural Network, the construction of a Regression Model, and the incorporation of a Random Forest Model. It involved data preprocessing, model construction, evaluation, and visualization of model performance.

### EDA

The exploratory data analysis process involved understanding the dataset's structure, distributions, missing values, and relationships between variables.

### Neural Network Implementation

Implemented a Neural Network using TensorFlow and Keras to predict house sale prices. The process included data preprocessing, hyperparameter tuning, model training, evaluation, and comparison with baseline predictions. Constructed a random forest model to complement the neural network, providing a comparative lens for analysis.

### Regression Model Construction

Constructed a regression model using a neural network architecture for house price prediction. This phase involved data retrieval, preprocessing, exploratory data analysis, model training, and evaluation. The random forest offered a comparative perspective on the analysis.

### Insights and Challenges

- **Key Insights**: Understanding data relationships, model performance evaluation, and visualization impact were crucial insights. Additionally, exploring the trade-offs between neural networks and random forest regression models provided valuable perspectives.
- **Challenges**: Addressed challenges including hyperparameter tuning and feature engineering to enhance predictive accuracy.

### Conclusion

The project provided insights into house price prediction using data-driven approaches. Exploratory analysis, model implementation, and evaluation facilitated a comprehensive understanding of the dataset and model performance. Challenges, including hyperparameter tuning and feature engineering, were addressed to improve predictive accuracy. The comparison between neural network and random forest models offered valuable perspectives on the strengths and trade-offs of each approach.

## Usage

Restart the kernel and run `regression_models.ipynb` to output the results of both a neural network and a random forest model.

The primary notebooks and documents include:

- `regression_models.ipynb`: Contains the final models
- `exploratory_data_analysis.ipynb`: Presents multiple plots displaying the data, guiding feature engineering, data cleaning, and processing
- `nn_template.ipynb`: Illustrates the usage of Keras Tuner for hyperparameter tuning in trial neural network models
- `House Prices Spreadsheet.xlsx`: Documentation of trial models and outcomes
- `/trial_models/`: Contains subsequent neural network and random forest models from trial iterations
- `predictive_models_report.md`: Summarizes results and discusses implications

## Gallery of Results

Note how closely both model predictions perform.

Sample Neural Network Evaluation Metrics:

![Picture of Sample Neural Network Evaluation Metrics](/images/nn_results.png)

![Sample Neural Network Evaluation Metrics](/images/nn_residuals_hist.png)

**Economic Implication For Histogram**: The diagram above highlights market segmentation, showing the number of properties that fall within certain price ranges. We can see that a good number of the houses fall within lower price ranges. Then reduce in number as they move away from the centermost point.

Neural Network Actual vs. Predicted Sale Price:

![Plot of Neural Network Actual vs. Predicted Sale Price](/images/nn_scatter.png)

**Economic Implication**: The diagram shows that actual and predicted prices are positively correlated or better still have a direct relationship. This implies that any discrepancies between the prices under measure might imply market inefficiencies or could indicate potential errors in our model.

Neural Network Training and Validation Loss:

![Plot of Neural Network Training and Validation Loss](/images/nn_loss.png)

**Economic Implication**: A decreasing trend in both training and validation loss indicates the model learns and generalizes well.

Neural Network Residuals:

![Plot of Neural Network Residuals](/images/nn_residuals.png)

**Economic Implication**: If the model did not provide a particular pattern, this implies our model did a great job accounting for systematic errors, which gives our results some validity. Random distribution around zero indicates the models capture most economic factors impacting sale prices.

Sample Random Forest Evaluation Metrics:

![Picture of Sample Random Forest Evaluation Metrics](/images/rf_results.png)

Random Forest Actual vs. Predicted Sale Price:

![Random Forest Actual vs. Predicted Sale Price](/images/rf_scatter.png)

**Economic Implication**: The diagram shows that actual and predicted prices are positively correlated or better still have a direct relationship. This implies that any discrepancies between the prices under measure might imply market inefficiencies or could indicate potential errors in our model.

Random Forest Residuals:

![Plot of Random Forest Residuals](/images/rf_residuals.png)

**Economic Implication**: If the model did not provide a particular pattern, this implies our model did a great job accounting for systematic errors, which gives our results some validity. Random distribution around zero indicates the models capture most economic factors impacting sale prices.

## References

Dataset provided by [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Licenses

This project is licensed under the terms of the [GNU General Public License version 2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html).

## Acknowledgements

Thanks to Justin Bisal, James Newman, and Geronimo Perez for feedback and assistance

## Authors

Moussa Diop, Abdullah Jaura, Bryan Johns, Bolima Tafah, November, 2023
