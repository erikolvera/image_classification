import os

class DataLoader:
    def __init__(self):
        self.digit_train_labels = []
        self.digit_validate_labels = []
        self.digit_test_labels = []
        
        self.digit_train_images = []
        self.digit_valid_images = []
        self.digit_test_images = []
        
        self.digit_train_features = []
        
        self.face_train_labels = []
        self.face_validate_labels = []
        self.face_test_labels = []
        
        self.face_train_images = []
        self.face_valid_images = []
        self.face_test_images = []
        
        self.face_train_features = []

    def read_labels(self, labels_file):
        try:
            with open(labels_file, 'r') as f:
                return [int(line) for line in f.read().splitlines()]
        except FileNotFoundError:
            print(f"Error: File not found {labels_file}")
            return []

    def read_images(self, images_file, total_images):
        try:
            with open(images_file, 'r') as f:
                content = f.read().splitlines()
        except FileNotFoundError:
            print(f"Error: File not found {images_file}")
            return [], 0
        
        if total_images == 0:
            return [], 0

        total_lines = len(content)
        lines_per_image = total_lines // total_images
        images = []
        
        for i in range(0, total_lines, lines_per_image):
            image = content[i:i+lines_per_image]
            if len(image) == lines_per_image:
                images.append(image)
        return images, lines_per_image

    def load_digits(self, base_path="cs4346-data/digitdata"):
        print("Loading digits...")
        self.digit_train_labels = self.read_labels(os.path.join(base_path, "traininglabels"))
        self.digit_validate_labels = self.read_labels(os.path.join(base_path, "validationlabels"))
        self.digit_test_labels = self.read_labels(os.path.join(base_path, "testlabels"))
        
        self.digit_train_images, _ = self.read_images(os.path.join(base_path, "trainingimages"), len(self.digit_train_labels))
        self.digit_valid_images, _ = self.read_images(os.path.join(base_path, "validationimages"), len(self.digit_validate_labels))
        self.digit_test_images, _ = self.read_images(os.path.join(base_path, "testimages"), len(self.digit_test_labels))

        print("Extracting digit features...")
        self.digit_train_features = [self.extract_features(img) for img in self.digit_train_images]

    def load_faces(self, base_path="cs4346-data/facedata"):
        print("Loading faces...")
        self.face_train_labels = self.read_labels(os.path.join(base_path, "facedatatrainlabels"))
        self.face_validate_labels = self.read_labels(os.path.join(base_path, "facedatavalidationlabels"))
        self.face_test_labels = self.read_labels(os.path.join(base_path, "facedatatestlabels"))
        
        self.face_train_images, _ = self.read_images(os.path.join(base_path, "facedatatrain"), len(self.face_train_labels))
        self.face_valid_images, _ = self.read_images(os.path.join(base_path, "facedatavalidation"), len(self.face_validate_labels))
        self.face_test_images, _ = self.read_images(os.path.join(base_path, "facedatatest"), len(self.face_test_labels))
        
        print("Extracting face features...")
        self.face_train_features = [self.extract_features(img) for img in self.face_train_images]

    @staticmethod
    def extract_raw_pixel_features(image):
        raw_list = []
        for row in image:
            for char in row:
                raw_list.append(1 if char in ('#', '+') else 0)
        return raw_list

    @staticmethod
    def extract_count_features(image):
        whitespace = 0
        symbols = 0
        for row in image:
            for char in row:
                if char == ' ':
                    whitespace += 1
                elif char in ('#', '+'):
                    symbols += 1
        return [whitespace, symbols]

    @staticmethod
    def extract_features(image):
        return DataLoader.extract_raw_pixel_features(image) + DataLoader.extract_count_features(image)

"""Maintain backward compatibility for single functions if desired, 
but for this refactor we prefer using the class in main.py.
However, main.py uses extract_features standalone."""

def extract_features(image):
    return DataLoader.extract_features(image)

if __name__ == "__main__":
    loader = DataLoader()
    loader.load_digits()
    loader.load_faces()
    
    print(f"Digit Train Images: {len(loader.digit_train_images)}")
    print(f"Face Train Images: {len(loader.face_train_images)}")
