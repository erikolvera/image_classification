from extract_data import DataLoader
from naive_bayes import NaiveBayes
import time

def test_naive_bayes():
    print("=" * 60)
    print("NAIVE BAYES CLASSIFIER EVALUATION")
    print("=" * 60)
    
    loader = DataLoader()
    
    # Digits Loading
    print("\n[1/2] Loading dataset 'Digits'...")
    loader.load_digits()
    num_train_digits = len(loader.digit_train_images)
    num_val_digits = len(loader.digit_valid_images)
    digit_feats = len(loader.digit_train_features[0]) if loader.digit_train_features else 0
    print(f"      • Training samples: {num_train_digits:,}")
    print(f"      • Validation samples: {num_val_digits:,}")
    print(f"      • Feature extraction: Complete ({digit_feats} features/image)")
    
    # Faces Loading
    print("\n[2/2] Loading dataset 'Faces'...")
    loader.load_faces()
    num_train_faces = len(loader.face_train_images)
    num_val_faces = len(loader.face_valid_images)
    face_feats = len(loader.face_train_features[0]) if loader.face_train_features else 0
    print(f"      • Training samples: {num_train_faces:,}")
    print(f"      • Validation samples: {num_val_faces:,}")
    print(f"      • Feature extraction: Complete ({face_feats} features/image)")
    
    print("\n" + "-" * 60)
    print(">> TRAINING & EVALUATING MODELS")
    print("-" * 60)
    
    # --- Digits Evaluation ---
    nb_digits = NaiveBayes(num_classes=10)
    start = time.time()
    nb_digits.train(loader.digit_train_features, loader.digit_train_labels)
    digit_time = time.time() - start
    
    correct_digits = 0
    total_digits = len(loader.digit_valid_images)
    digit_valid_features = [loader.extract_features(img) for img in loader.digit_valid_images]
    
    for features, label in zip(digit_valid_features, loader.digit_validate_labels):
        if nb_digits.predict(features) == label:
            correct_digits += 1
            
    digit_acc = (correct_digits / total_digits) * 100
    
    print("\n[MODEL 1] Digits Classification (10 classes: 0-9)")
    print(f"      • Training time:  {digit_time:.2f} seconds")
    print(f"      • Validation:     {correct_digits} / {total_digits} correctly classified")
    print(f"      ► Accuracy:       {digit_acc:.2f}%")

    # --- Faces Evaluation ---
    nb_faces = NaiveBayes(num_classes=2)
    start = time.time()
    nb_faces.train(loader.face_train_features, loader.face_train_labels)
    face_time = time.time() - start
    
    correct_faces = 0
    total_faces = len(loader.face_valid_images)
    face_valid_features = [loader.extract_features(img) for img in loader.face_valid_images]
    
    for features, label in zip(face_valid_features, loader.face_validate_labels):
        if nb_faces.predict(features) == label:
            correct_faces += 1
            
    face_acc = (correct_faces / total_faces) * 100
    
    print("\n[MODEL 2] Faces Classification (2 classes: Face / Non-Face)")
    print(f"      • Training time:  {face_time:.2f} seconds")
    print(f"      • Validation:     {correct_faces} / {total_faces} correctly classified")
    print(f"      ► Accuracy:       {face_acc:.2f}%")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    test_naive_bayes()
