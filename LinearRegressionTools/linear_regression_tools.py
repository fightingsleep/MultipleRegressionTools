import argparse
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model

def build_linear_model(data_frame):
    independent_vars_values = data_frame[data_frame.columns[1:]]
    dependent_var_values = data_frame[data_frame.columns[0]]

    model_data = linear_model.LinearRegression().fit(independent_vars_values, dependent_var_values)
    return {'intercept': model_data.intercept_, 'model_data': model_data.coef_}

def visualize_linearity(data_frame):
    num_independent_vars = data_frame.shape[1]
    dependent_var_name = data_frame.columns[0]
    for i in range(1, num_independent_vars):
        independent_var_name = data_frame.columns[i]
        plt.scatter(data_frame[independent_var_name], data_frame[dependent_var_name])
        plt.title('{0} Vs {1}'.format(dependent_var_name, independent_var_name))
        plt.xlabel(independent_var_name)
        plt.ylabel(dependent_var_name)
        plt.show()

def main():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='<input file path>',
        help='input file containing the data for linear regression')
    parser.add_argument('-v', action='store_true', help='visualize the data')
    args = parser.parse_args()

    # Parse the csv file containing the regression data
    data_frame = pandas.read_csv(args.input).dropna()

    # Optionally display graphs showing relationships between the
    # independent variables and the dependent variable
    if args.v:
        visualize_linearity(data_frame)

    # Build the statistical model from the input data
    build_linear_model(data_frame)

    # Allow the user to make predictions using the model
    #make_predictions(model_data)

if __name__ == "__main__":
    main()
