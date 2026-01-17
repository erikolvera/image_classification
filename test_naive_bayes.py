from extract_data import DataLoader
from naive_bayes import NaiveBayes
import time

def test_naive_bayes():
    print("Initializing DataLoader...")
    loader = DataLoader()
    loader.load_digits()
    loader.load_faces()
    
    # --- Digits ---
    print("\nTraining Naive Bayes on Digits...")
    nb_digits = NaiveBayes(num_classes=10)
    start = time.time()
    nb_digits.train(loader.digit_train_features, loader.digit_train_labels)
    print(f"Training took {time.time() - start:.2f}s")
    
    print("Evaluating on Validation set...")
    correct = 0
    total = len(loader.digit_valid_images)
    digit_valid_features = [loader.extract_features(img) for img in loader.digit_valid_images]
    
    for features, label in zip(digit_valid_features, loader.digit_validate_labels):
        if nb_digits.predict(features) == label:
            correct += 1
            
    print(f"Naive Bayes Digits Accuracy: {(correct/total)*100:.2f}%")

    # --- Faces ---
    print("\nTraining Naive Bayes on Faces...")
    nb_faces = NaiveBayes(num_classes=2)
    start = time.time()
    nb_faces.train(loader.face_train_features, loader.face_train_labels)
    print(f"Training took {time.time() - start:.2f}s")
    
    print("Evaluating on Validation set...")
    correct = 0
    total = len(loader.face_valid_images)
    face_valid_features = [loader.extract_features(img) for img in loader.face_valid_images]
    
    for features, label in zip(face_valid_features, loader.face_validate_labels):
        if nb_faces.predict(features) == label:
            correct += 1
            
    print(f"Naive Bayes Faces Accuracy: {(correct/total)*100:.2f}%")

if __name__ == "__main__":
    test_naive_bayes()
