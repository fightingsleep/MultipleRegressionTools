import unittest
import linear_regression_tools as lrt

class TestLinearRegression(unittest.TestCase):
    def test_basic_prediction(self):
        regression = lrt.RegressionOrchestrator()
        regression.initialize_model(r"..\data.csv")
        self.assertGreater(regression.make_prediction([1,1,1,1,1,1,1,1]), 0)

    def test_input_analyzer(self):
        analyzer = lrt.RegressionInputAnalyzer(r"..\data.csv")
        self.assertIsNotNone(analyzer.calculate_statistics('Tax', 'Bathroom == 2 & Bedroom == 4'))

if __name__ == '__main__':
    unittest.main()
