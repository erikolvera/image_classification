import unittest
from extract_data import DataLoader, extract_features
from naive_bayes import NaiveBayes

class TestRefactor(unittest.TestCase):
    def test_feature_extraction(self):
        # Mock image: 2x2
        image = [
            " #",
            "+ "
        ]
        # raw: ' '->0, '#'->1, '+'->1, ' '->0
        # flattened: [0, 1, 1, 0]
        expected_raw = [0, 1, 1, 0]
        
        # count: whitespace=2, symbols=2
        expected_count = [2, 2]
        
        features = extract_features(image)
        self.assertEqual(features, expected_raw + expected_count)

    def test_naive_bayes_mock(self):
        # 2 classes, 1 feature
        nb = NaiveBayes(num_classes=2)
        
        # Train data
        # Class 0: feature=0
        # Class 1: feature=1
        features = [[0], [1], [1], [1]]
        labels = [0, 1, 1, 1]
        
        nb.train(features, labels)
        
        # Predict
        pred0 = nb.predict([0])
        pred1 = nb.predict([1])
        
        self.assertEqual(pred0, 0)
        self.assertEqual(pred1, 1)

if __name__ == '__main__':
    unittest.main()
