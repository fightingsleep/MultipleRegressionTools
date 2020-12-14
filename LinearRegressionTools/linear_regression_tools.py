import argparse
from collections import OrderedDict
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model

class RegressionOrchestrator:
    def __init__(self):
        self.data_frame = None
        self.linear_regression = linear_model.LinearRegression()
        self.model_params = OrderedDict()

    def initialize_model(self, path: str) -> None:
        # Parse the csv file containing the regression data
        self.data_frame = pandas.read_csv(path).dropna()

        independent_vars_values = self.data_frame[self.data_frame.columns[1:]]
        dependent_var_values = self.data_frame[self.data_frame.columns[0]]

        # Perform regression
        regression_output = self.linear_regression.fit(
            independent_vars_values, dependent_var_values)

        # Set the model parameters
        self.model_params['Intercept'] = regression_output.intercept_
        for i in range(1, self.data_frame.shape[1]):
            self.model_params[self.data_frame.columns[i]] = regression_output.coef_[i - 1]

    def make_prediction(self, independent_vars_values: list) -> list:
        return self.linear_regression.predict(independent_vars_values)

    def get_model_parameters_string(self) -> str:
        output_string = 'Model parameters:\n'
        for key in self.model_params:
            output_string += "  {0} : {1}\n".format(key, self.model_params[key])
        return output_string

class RegressionInputAnalyzer:
    def __init__(self, path: str):
        self.data_frame = pandas.read_csv(path).dropna()

    def visualize_linearity(self) -> None:
        num_independent_vars = self.data_frame.shape[1]
        dependent_var_name = self.data_frame.columns[0]
        for i in range(1, num_independent_vars):
            independent_var_name = self.data_frame.columns[i]
            plt.scatter(self.data_frame[independent_var_name], self.data_frame[dependent_var_name])
            plt.title('{0} Vs {1}'.format(dependent_var_name, independent_var_name))
            plt.xlabel(independent_var_name)
            plt.ylabel(dependent_var_name)
            plt.show()

    def calculate_statistics(self, variable_name: str, filter_string: str = None) -> dict:
        df = self.data_frame
        if filter_string is not None:
            df = df.query(filter_string)
        stats = df[variable_name].describe().to_dict()
        stats['median'] = df[variable_name].median()
        return stats

def main():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='<input file path>',
        help='input file containing the data for linear regression')
    parser.add_argument('-v', action='store_true', help='visualize the data')
    args = parser.parse_args()

    # Optionally display graphs showing relationships between the
    # independent variables and the dependent variable
    if args.v:
        input_analyzer = RegressionInputAnalyzer(args.input)
        input_analyzer.visualize_linearity()

    # Initialize the statistical model from the input data
    regression = RegressionOrchestrator()
    regression.initialize_model(args.input)
    print(regression.get_model_parameters_string())

    # Allow the user to make a prediction using the model
    while True:
        try:
            independent_vars = [float(x) for x in input(
                "Enter independent variable values (ex: 1 1 1 1 1 1 1 1): ").split()]
            print("Predicted price: ", regression.make_prediction([independent_vars]))
        except:
            print("invalid input")

if __name__ == "__main__":
    main()
