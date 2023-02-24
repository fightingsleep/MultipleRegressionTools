import argparse
from collections import OrderedDict
import pandas
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

class RegressionOrchestrator:
    """A simple class for performing multiple regression"""
    def __init__(self):
        self.data_frame = None
        self.linear_regression = lm.LinearRegression()
        self.model_params = OrderedDict()

    def initialize_model(self, path: str) -> None:
        """Setup the regression model so it can be used for predictions"""
        # Parse the csv file containing the regression data
        self.data_frame = pandas.read_csv(path).dropna()

        independent_var_names = self.data_frame.columns[1:]
        dependent_var_name = self.data_frame.columns[0]
        independent_vars_values = self.data_frame[independent_var_names]
        dependent_var_values = self.data_frame[dependent_var_name]

        # Perform regression
        regression_output = self.linear_regression.fit(
            independent_vars_values, dependent_var_values)

        # Set the model parameters
        self.model_params['Intercept'] = regression_output.intercept_
        for i in range(0, len(independent_var_names)):
            self.model_params[independent_var_names[i]] = regression_output.coef_[i]

    def make_prediction(self, independent_vars_values: list) -> list:
        """Use the initialized regression model to make a prediction"""
        return self.linear_regression.predict(independent_vars_values)

    def get_model_parameters_string(self) -> str:
        """Constructs a pretty-string containing the regression model parameters"""
        output_string = 'Model parameters:\n'
        for key in self.model_params:
            output_string += "  {0} : {1}\n".format(key, self.model_params[key])
        return output_string

    def visualize_regression(self) -> None:
        """Visualizes the regression models line/plane of best fit for 2D/3D input data"""
        dimensions = self.data_frame.shape[1]
        # Can't visualize anything over 3 dimensions
        if dimensions > 3:
            return
        if dimensions == 2:
            dependent_var_name = self.data_frame.columns[0]
            independent_var_name = self.data_frame.columns[1]
            # Scatter plot the data that was input by the user
            plt.scatter(self.data_frame[independent_var_name], self.data_frame[dependent_var_name])
            i_min = int(self.data_frame[independent_var_name].min())
            i_max = int(self.data_frame[independent_var_name].max())
            d_height = int(self.data_frame[dependent_var_name].max())
            x = np.linspace(i_min,i_max,d_height)
            y = self.model_params[independent_var_name] * x + self.model_params['Intercept']
            # Plot the regression line through the scattered data
            plt.plot(x, y, color='red')
            plt.title('{0} Vs {1}'.format(dependent_var_name, independent_var_name))
            plt.xlabel(independent_var_name)
            plt.ylabel(dependent_var_name)
            plt.show()
        if dimensions == 3:
            dependent_var_name = self.data_frame.columns[0]
            first_independent_var_name = self.data_frame.columns[1]
            second_independent_var_name = self.data_frame.columns[2]

            # Scatter plot the user input data in 3d space. 
            fig = plt.figure(figsize=[16, 9])
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.scatter(
                self.data_frame[first_independent_var_name],
                self.data_frame[second_independent_var_name],
                self.data_frame[dependent_var_name])

            # Figure out the dimensions of the input data
            min1 = int(self.data_frame[first_independent_var_name].min())
            max1 = int(self.data_frame[first_independent_var_name].max())
            min2 = int(self.data_frame[second_independent_var_name].min())
            max2 = int(self.data_frame[second_independent_var_name].max())
            height = int(self.data_frame[dependent_var_name].max())

            # Plot the plane of best fit through the scattered data
            X,Y = np.meshgrid(np.linspace(min1,max1,height), np.linspace(min2,max2,height))
            Z = self.model_params['Intercept'] + self.model_params[first_independent_var_name] * X + self.model_params[second_independent_var_name] * Y

            ax.plot_surface(X, Y, Z, cmap='rainbow', alpha=0.75)
            ax.set_title('{0} Vs {1} & {2}'.format(dependent_var_name, first_independent_var_name, second_independent_var_name), pad=20)
            ax.set_xlabel(first_independent_var_name,labelpad=20)
            ax.set_ylabel(second_independent_var_name,labelpad=20)
            ax.set_zlabel(dependent_var_name, labelpad=20)
            plt.show()



class RegressionInputAnalyzer:
    """A simple class for analyzing data before performing regression on it"""
    def __init__(self, path: str):
        self.data_frame = pandas.read_csv(path).dropna()

    def visualize_linearity(self) -> None:
        """A method for visualizing linearity between a single independent variable and the dependent variable"""
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
        """Calculate a bunch of statistics for a given regression variable"""
        df = self.data_frame
        if filter_string is not None:
            df = df.query(filter_string)
        stats = df[variable_name].describe().to_dict()
        stats['median'] = df[variable_name].median()
        return stats

def main():
    """Main functions providing an example of basic usage of this module"""
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

    if args.v:
        # Attempt to visualize the line/plane of best fit
        regression.visualize_regression()

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
