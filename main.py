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







