# Naive Bayes Classifier
"""
Features = The input data = What the classifier SEES
Labels = The "answer key" = What the classifier SHOULD predict
"""
class NaiveBayes:
    def __init__(self, num_classes):
        self.num_classes = num_classes  # 10 for digits, 2 for faces
        self.class_counts = {}
        self.feature_counts = {}
        self.total_samples = 0

        def train(self, features_list, labels):
            for cls in range(self.num_classes):
                self.class_counts[cls] = 0
                self.feature_counts[cls] = {}

            self.total_samples = len(labels)

            for features, label in zip(features_list,labels):
                self.class_counts[label] += 1

                for feature_index, feature_value in enumerate(features):
                    if feature_index not in self.feature_counts[label]:
                        self.feature_counts[label][feature_index] = {}
                    if feature_value not in self.feature_counts[label][feature_index]:
                        self.feature_counts[label][feature_index][feature_value] = 0
                    """feature_counts[5][42][1] = 380
                    Translation: "Out of all the digit '5' images I saw, feature position #42 had the value 1 (was lit/foreground) in 380 of them"""
                    self.feature_counts[label][feature_index][feature_value] += 1