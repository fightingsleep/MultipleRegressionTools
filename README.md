# LinearRegressionTools
This python module allows the user to easily perform linear regression

It contains two main classes:
    - RegressionOrchestrator
    - RegressionInputAnalyzer
    
Both of these classes take a path to a csv file. The file must be setup such that the first column contains the dependent variable values, and the remaining columns are independent variable values. The first row in each column represents the names of the variables. See the data.csv file for an example of the formatting.

The RegressionInputAnalyzer class allows you to visualize the linearity between the dependent and independent variable values. It also allows you to obtain a set of statistics for a given variable.

The RegressionOrchestrator is responsible for performing the actual regression. Simply initialize the model and then make predictions using the model. The user also has access to the intercept and coefficients that were calculated by the model. These parameters are solved using the ordinary least squares method. See https://en.wikipedia.org/wiki/Ordinary_least_squares for details.




