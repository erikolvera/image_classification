# train -> validate -> test

# digits
#5000 lines of digits training labels
#140000 lines of digits training images
# total height = 140000/5000 = 28 lines per image

# faces
#451 lines face data training labels
# 31570 lines of face data train
# so total height = 31570 / 451 = 70 lines per face


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

    height = total_lines // total_images # // produces an integer instead of a float
    images = []

    for i in range(0, total_lines, height):
        one_image = images_content[i:i+height]
        if len(one_image) == height:
            images.append(one_image)
    return images, height

# digits
digit_train_labels =read_labels("cs4346-data/digitdata/traininglabels")
digit_validate_labels = read_labels("cs4346-data/digitdata/validationlabels")
digit_test_labels = read_labels("cs4346-data/digitdata/testlabels")

digit_train_images, digit_height = read_images("cs4346-data/digitdata/trainingimages", total_images=len(digit_train_labels))
digit_valid_images, _ = read_images("cs4346-data/digitdata/validationimages", total_images=len(digit_validate_labels))
digit_test_images, _ = read_images("cs4346-data/digitdata/testimages",total_images=len(digit_test_labels))

# faces

face_train_labels = read_labels("cs4346-data/facedata/facedatatrainlabels")
face_validate_labels = read_labels("cs4346-data/facedata/facedatavalidationlabels")
face_test_labels  = read_labels("cs4346-data/facedata/facedatatestlabels")

face_train_images, image_height = read_images("cs4346-data/facedata/facedatatrain", total_images=len(face_train_labels))
face_valid_images, _ = read_images("cs4346-data/facedata/facedatavalidation", total_images=len(face_validate_labels))
face_test_images, _ = read_images("cs4346-data/facedata/facedatatest", total_images=len(face_test_labels))


# making sure code actually works
# print(digit_height)
# print(len(digit_train_images), len(digit_train_labels))
#
# print(image_height)
# print(len(face_train_images),len(face_train_labels))
