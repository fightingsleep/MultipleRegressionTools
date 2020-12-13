import argparse
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model
from collections import OrderedDict

class RegressionOrchestrator:
    def __init__(self, df):
        self.data_frame = df
        self.linear_regression = linear_model.LinearRegression()
        self.model_params = OrderedDict()

    def initialize_model(self):
        independent_vars_values = self.data_frame[self.data_frame.columns[1:]]
        dependent_var_values = self.data_frame[self.data_frame.columns[0]]

        # Perform regression
        regression_output = self.linear_regression.fit(independent_vars_values, dependent_var_values)

        # Set the model parameters
        self.model_params['Intercept'] = regression_output.intercept_
        for x in range(1, self.data_frame.shape[1]):
            self.model_params[self.data_frame.columns[x]] = regression_output.coef_[x - 1]

    def make_prediction(self, independent_vars_values):
        return self.linear_regression.predict([independent_vars_values])

    def visualize_linearity(self):
        num_independent_vars = self.data_frame.shape[1]
        dependent_var_name = self.data_frame.columns[0]
        for i in range(1, num_independent_vars):
            independent_var_name = self.data_frame.columns[i]
            plt.scatter(self.data_frame[independent_var_name], self.data_frame[dependent_var_name])
            plt.title('{0} Vs {1}'.format(dependent_var_name, independent_var_name))
            plt.xlabel(independent_var_name)
            plt.ylabel(dependent_var_name)
            plt.show()

    def get_model_parameters(self):
        return self.model_params

def main():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='<input file path>',
        help='input file containing the data for linear regression')
    parser.add_argument('-v', action='store_true', help='visualize the data')
    args = parser.parse_args()

    # Parse the csv file containing the regression data
    data_frame = pandas.read_csv(args.input).dropna()

    # Initialize the statistical model from the input data
    regression = RegressionOrchestrator(data_frame)
    regression.initialize_model()

    # Optionally display graphs showing relationships between the
    # independent variables and the dependent variable
    if args.v:
        regression.visualize_linearity()

    # Allow the user to make a prediction using the model
    print(regression.make_prediction([1,1,1,1,1,1,1,1]))

    print(regression.get_model_parameters())

if __name__ == "__main__":
    main()
