"""step 1: data processing

train -> validate -> test

DIGITS:
5000 lines of digits training labels
140000 lines of digits training images
total lines_per_image = 140000/5000 = 28 lines per image

FACES:
451 lines face data training labels
31570 lines of face data train
so total lines_per_image = 31570 / 451 = 70 lines per face """

import math
import time
import random
import statistics # Not used yet but might be useful later

#wanna read the labels first. theyre basic integers
def read_labels(labels_file):
    labels = []
    with open(labels_file, 'r') as f:
        labels_content = f.read().splitlines()
        for line in labels_content:
            labels.append(int(line))
    return labels

# want to read the images
def read_images(images_files, total_images):
    with open(images_files, 'r') as images_f:
        images_content = images_f.read().splitlines()

    total_lines = len(images_content)

    lines_per_image = total_lines // total_images
    images = []

    for i in range(0, total_lines, lines_per_image):
        one_image = images_content[i:i+lines_per_image] # extract the full height of the image starting at i
        if len(one_image) == lines_per_image:
            images.append(one_image)
    return images, lines_per_image

# digits
digit_train_labels =read_labels("cs4346-data/digitdata/traininglabels")
digit_validate_labels = read_labels("cs4346-data/digitdata/validationlabels")
digit_test_labels = read_labels("cs4346-data/digitdata/testlabels")

digit_train_images, digit_lines_per_image = read_images("cs4346-data/digitdata/trainingimages", total_images=len(digit_train_labels))
digit_valid_images, _ = read_images("cs4346-data/digitdata/validationimages", total_images=len(digit_validate_labels))
digit_test_images, _ = read_images("cs4346-data/digitdata/testimages",total_images=len(digit_test_labels))

# faces

face_train_labels = read_labels("cs4346-data/facedata/facedatatrainlabels")
face_validate_labels = read_labels("cs4346-data/facedata/facedatavalidationlabels")
face_test_labels  = read_labels("cs4346-data/facedata/facedatatestlabels")

face_train_images, image_lines_per_image = read_images("cs4346-data/facedata/facedatatrain", total_images=len(face_train_labels))
face_valid_images, _ = read_images("cs4346-data/facedata/facedatavalidation", total_images=len(face_validate_labels))
face_test_images, _ = read_images("cs4346-data/facedata/facedatatest", total_images=len(face_test_labels))


# making sure code actually works
# print(digit_lines_per_image)
# print(len(digit_train_images), len(digit_train_labels))
#
# print(image_lines_per_image)
# print(len(face_train_images),len(face_train_labels))

def extract_raw_pixel_features(image):
    raw_list = []
    for row in image:
        for char in row:
            if char == '#' or char == '+':
                raw_list.append(1)
            else:
                raw_list.append(0)
    return raw_list

def extract_count_features(image):
    whitespace = 0
    symbols = 0

    for row in image:
        for char in row:
            if char == ' ':
                whitespace += 1
            elif char == '#' or char =='+':
                symbols += 1
            else:
                pass
    return [whitespace, symbols]

# Added a function to combine both feature extraction methods 
def extract_features(image):
    raw = extract_raw_pixel_features(image)
    count = extract_count_features(image)
    return raw + count

digit_raw_features = [extract_raw_pixel_features(image) for image in digit_train_images]
digit_count_features = [extract_count_features(image) for image in digit_train_images]

face_raw_features = [extract_raw_pixel_features(image) for image in face_train_images]
face_count_features = [extract_count_features(image) for image in face_train_images]

# combine raw pixel features and count features into one feature vector per training image
digit_train_features = [
    digit_raw_features[i] + digit_count_features[i]
    for i in range(len(digit_train_images))
]

# combine raw pixel features and count features into one feature vector per training image
face_train_features = [
    face_raw_features[i] + face_count_features[i]
    for i in range(len(face_train_images))
]

# Debugging quick checks, can be removed later before submsion 
if __name__ == "__main__":

    # Basic size checks
    print("DIGITS:")
    print("  train images:", len(digit_train_images))
    print("  train labels:", len(digit_train_labels))
    print("  valid images:", len(digit_valid_images))
    print("  valid labels:", len(digit_validate_labels))
    print("  test  images:", len(digit_test_images))
    print("  test  labels:", len(digit_test_labels))
    print("  lines per digit image:", digit_lines_per_image)

    print("\nFACES:")
    print("  train images:", len(face_train_images))
    print("  train labels:", len(face_train_labels))
    print("  valid images:", len(face_valid_images))
    print("  valid labels:", len(face_validate_labels))
    print("  test  images:", len(face_test_images))
    print("  test  labels:", len(face_test_labels))
    print("  lines per face image:", image_lines_per_image)

    # Feature vector length checks
    print("\nFEATURE SHAPES:")
    print("  digit_train_features: ",
          len(digit_train_features), "x", len(digit_train_features[0]))
    print("  face_train_features:  ",
          len(face_train_features), "x", len(face_train_features[0]))

    # Quick peek at one image & features
    print("\nExample digit label:", digit_train_labels[0])
    print("Example digit image:")
    for row in digit_train_images[0]:
        print(row)

    print("\nRaw pixel feature vector (first 50 entries):")
    print(digit_raw_features[0][:50])

    print("\nCount features [whitespace, symbols]:")
    print(digit_count_features[0])



# Begin Classfication Algorithms

# Naive Bayes Classifier
class NaiveBayes:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.class_counts = {}
        self.feature_counts = {}
        self.total_samples = 0

    # Train Naive Bayes Classifier    
    def train(self, features_list, labels):
        # Initialize counts
        for c in range(self.num_classes):
            self.class_counts[c] = 0
            self.feature_counts[c] = {}
        
        # Count occurrences of classes and features
        self.total_samples = len(labels)
        num_features = len(features_list[0])
        
        for features, label in zip(features_list, labels):
            self.class_counts[label] += 1
            
            for feature_idx, feature_val in enumerate(features):
                if feature_idx not in self.feature_counts[label]:
                    self.feature_counts[label][feature_idx] = {0: 0, 1: 0}
                
                # For binary features 
                if feature_val == 0 or feature_val == 1:
                    self.feature_counts[label][feature_idx][feature_val] += 1
                else:
                    # For count features (continuous), we'll discretize
                    if feature_val not in self.feature_counts[label][feature_idx]:
                        self.feature_counts[label][feature_idx][feature_val] = 0
                    self.feature_counts[label][feature_idx][feature_val] += 1
    
    # Predict class for given features using Naive Bayes
    def predict(self, features):
        best_class = None
        best_prob = float('-inf')
        
        for c in range(self.num_classes):
            # Start with log prior probability
            log_prob = math.log((self.class_counts[c] + 1) / (self.total_samples + self.num_classes))
            
            # Add log likelihood for each feature
            for feature_idx, feature_val in enumerate(features):
                if feature_idx in self.feature_counts[c]:
                    # Laplace smoothing
                    if feature_val in self.feature_counts[c][feature_idx]:
                        count = self.feature_counts[c][feature_idx][feature_val]
                    else:
                        count = 0
                    
                    total_count = sum(self.feature_counts[c][feature_idx].values())
                    num_values = len(self.feature_counts[c][feature_idx])
                    
                    prob = (count + 1) / (total_count + num_values)
                    log_prob += math.log(prob)
            
            if log_prob > best_prob:
                best_prob = log_prob
                best_class = c
        
        return best_class

# Perceptron Classifier
class Perceptron:
    def __init__(self, num_classes, num_features, learning_rate=0.1, epochs=10):
        self.num_classes = num_classes
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Initialize weights for each class
        self.weights = {c: [0.0] * num_features for c in range(num_classes)}
        self.bias = {c: 0.0 for c in range(num_classes)}
    
    #Train Perceptron Classifier
    def train(self, features_list, labels):
        for epoch in range(self.epochs):
            for features, label in zip(features_list, labels):
                # Predict
                predicted = self.predict(features)
                
                # Update weights if prediction is wrong
                if predicted != label:
                    # Increase weights for correct class
                    for i in range(self.num_features):
                        self.weights[label][i] += self.learning_rate * features[i]
                    self.bias[label] += self.learning_rate
                    
                    # Decrease weights for predicted class
                    for i in range(self.num_features):
                        self.weights[predicted][i] -= self.learning_rate * features[i]
                    self.bias[predicted] -= self.learning_rate
    
    # Predict class for given features
    def predict(self, features):
        best_class = None
        best_score = float('-inf')
        
        for c in range(self.num_classes):
            score = self.bias[c]
            for i in range(self.num_features):
                score += self.weights[c][i] * features[i]
            
            if score > best_score:
                best_score = score
                best_class = c
        
        return best_class

# Performance Evaluation that classifies with random sampling. Returns mean accuracy, std deviation of accuracy, and mean runtime
def evaluate_classifier(classifier_class, train_images, train_labels, test_images, 
                        test_labels, num_classes, percentage, num_iterations=5, **kwargs):
    accuracies = []
    runtimes = []
    for iteration in range(num_iterations):
        # Randomly sample training data
        num_samples = int(len(train_images) * percentage / 100)
        indices = random.sample(range(len(train_images)), num_samples)
        
        sampled_images = [train_images[i] for i in indices]
        sampled_labels = [train_labels[i] for i in indices]
        
        # Extract features
        sampled_features = [extract_features(img) for img in sampled_images]
        test_features = [extract_features(img) for img in test_images]
        
        # Train classifier
        start_time = time.time()
        if classifier_class == NaiveBayes:
            classifier = NaiveBayes(num_classes)
        else:  # Perceptron
            classifier = Perceptron(num_classes, len(sampled_features[0]), **kwargs)
        
        classifier.train(sampled_features, sampled_labels)
        runtime = time.time() - start_time
        
        # Test classifier
        correct = 0
        for features, label in zip(test_features, test_labels):
            if classifier.predict(features) == label:
                correct += 1
        
        accuracy = (correct / len(test_labels)) * 100
        accuracies.append(accuracy)
        runtimes.append(runtime)
    
    # Calculate mean and std deviation of accuracy and mean runtime
    mean_acc = sum(accuracies) / len(accuracies)
    variance = sum((x - mean_acc) ** 2 for x in accuracies) / len(accuracies)
    std_acc = math.sqrt(variance)
    mean_runtime = sum(runtimes) / len(runtimes)
    
    return mean_acc, std_acc, mean_runtime

# Experimentation and Reporting
def run_experiments(dataset_name, train_images, train_labels, test_images, test_labels, num_classes):
    print(f"\n{'='*80}")
    print(f"EXPERIMENTS ON {dataset_name.upper()} DATASET")
    print(f"{'='*80}\n")
    
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    nb_results = []
    perceptron_results = []
    
    for pct in percentages:
        print(f"Training with {pct}% of data...")
        
        # Naive Bayes
        nb_acc, nb_std, nb_time = evaluate_classifier(
            NaiveBayes, train_images, train_labels, test_images, test_labels, 
            num_classes, pct, num_iterations=5
        )
        nb_results.append((pct, nb_acc, nb_std, nb_time))
        
        # Perceptron
        p_acc, p_std, p_time = evaluate_classifier(
            Perceptron, train_images, train_labels, test_images, test_labels, 
            num_classes, pct, num_iterations=5, learning_rate=0.1, epochs=10
        )
        perceptron_results.append((pct, p_acc, p_std, p_time))
        
        # Print results 
        print(f"  Naive Bayes - Accuracy: {nb_acc:.2f}% (±{nb_std:.2f}), Time: {nb_time:.4f}s")
        print(f"  Perceptron  - Accuracy: {p_acc:.2f}% (±{p_std:.2f}), Time: {p_time:.4f}s")
    
    # Print summary table
    print(f"\n{'-'*80}")
    print(f"SUMMARY TABLE - {dataset_name.upper()}")
    print(f"{'-'*80}")
    print(f"{'Data%':<8}{'NB Acc':<12}{'NB Std':<12}{'NB Time':<12}{'P Acc':<12}{'P Std':<12}{'P Time':<12}")
    print(f"{'-'*80}")
    
    for i, pct in enumerate(percentages):
        nb_pct, nb_acc, nb_std, nb_time = nb_results[i]
        p_pct, p_acc, p_std, p_time = perceptron_results[i]
        print(f"{pct:<8}{nb_acc:<12.2f}{nb_std:<12.2f}{nb_time:<12.4f}{p_acc:<12.2f}{p_std:<12.2f}{p_time:<12.4f}")
    
    return nb_results, perceptron_results

# MAIN EXECUTION

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CS 4346: Image Classification Experiments")
    print("="*80)
    
    # Run experiments on the digit dataset (10 classes 0-9)
    digit_nb_results, digit_p_results = run_experiments(
        "DIGIT", digit_train_images, digit_train_labels, 
        digit_test_images, digit_test_labels, num_classes=10
    )
    
    # Run experiments on the face dataset (2 classes 0=not face, 1=face)
    face_nb_results, face_p_results = run_experiments(
        "FACE", face_train_images, face_train_labels, 
        face_test_images, face_test_labels, num_classes=2
    )
    
    # Final message
    print("\n" + "="*80)
    print("ALL EXPERIMENTATION COMPLETED!")
    print("="*80 + "\n")
