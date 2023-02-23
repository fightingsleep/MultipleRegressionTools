import unittest
from multiple_regression_tools import multiple_regression as mr

class TestLinearRegression(unittest.TestCase):
    def test_basic_prediction(self):
        regression = mr.RegressionOrchestrator()
        regression.initialize_model("./tests/data.csv")
        self.assertGreater(regression.make_prediction([[1,1,1,1,1,1,1,1]]), 0)

    def test_input_analyzer(self):
        analyzer = mr.RegressionInputAnalyzer("./tests/data.csv")
        self.assertIsNotNone(analyzer.calculate_statistics('Tax', 'Bathroom == 2 & Bedroom == 4'))

if __name__ == '__main__':
    unittest.main()
