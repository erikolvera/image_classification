# train -> validate -> test

# digits
#5000 lines of digits training labels
#140000 lines of digits training images
# total height = 14000/5000 = 28 lines per image

# faces
#451 lines face data training labels
# 31570 lines of face data train
# so total height = 31580 / 451 = 70 lines per face



#wanna read the labels first. theyre basic integers
def read_labels(labels_file):
    labels = []
    with open(labels_file, 'r') as f:
        labels_content = f.read().splitlines()
        for line in labels_content:
            labels.append(int(line))

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
    return images


