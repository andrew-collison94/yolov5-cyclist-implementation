import pandas as pd
import cv2
import os
import random
import matplotlib.pyplot as plt
from .hyperparameters import num_samples, train_epochs, train_images_path, train_labels_path, runs_directory, results_column, hyp_iou_files


def display_sample(num_samples, images_path, labels_path):
    """
    Display selection of sample of images with bounding boxes
    Adapted from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
    via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
    """

    # Get all image files
    image_files = os.listdir(images_path)

    # Randomly select images that have label files
    sample_images = []
    for _ in range(num_samples):
        random_image_file = random.choice(image_files)
        label_file = random_image_file[:-3] + 'txt'
        label_path = os.path.join(labels_path, label_file)

        while not os.path.exists(label_path):
            random_image_file = random.choice(image_files)
            label_file = random_image_file[:-3] + 'txt'
            label_path = os.path.join(labels_path, label_file)

        sample_images.append(random_image_file)

    # Display bounding boxes on images
    # Adjusting figure size based on num_samples
    fig, axs = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))

    for ax, img_name in zip(axs, sample_images):
        img_path = os.path.join(images_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        dh, dw, _ = img.shape

        label_name = img_name[:-3] + 'txt'
        label_path = os.path.join(labels_path, label_name)

        with open(label_path, 'r') as f:
            data = f.readlines()
            for dt in data:
                _, x, y, w, h = map(float, dt.split(' '))
                l = int((x - w / 2) * dw)
                r = int((x + w / 2) * dw)
                t = int((y - h / 2) * dh)
                b = int((y + h / 2) * dh)

                cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)

        ax.imshow(img)
        ax.set_title(img_name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def dataset_distribution(train_images_path, valid_images_path, test_images_path):
    train_images_count = len(os.listdir(train_images_path))
    valid_images_count = len(os.listdir(valid_images_path))
    test_images_count = len(os.listdir(test_images_path))

    plt.pie([train_images_count, valid_images_count, test_images_count], labels=[
            'Train', 'Validation', 'Test'], autopct='%1.1f%%')
    plt.title('Distribution of Train/Validation/Test Split')
    plt.show()


def background_split(train_labels_path, valid_labels_path, test_labels_path):
    # Counting the total number of label files and the number of empty label files in each directory
    train_labels_count = len(os.listdir(train_labels_path))
    train_empty_labels_count = len([name for name in os.listdir(
        train_labels_path) if os.path.getsize(os.path.join(train_labels_path, name)) == 0])

    valid_labels_count = len(os.listdir(valid_labels_path))
    valid_empty_labels_count = len([name for name in os.listdir(
        valid_labels_path) if os.path.getsize(os.path.join(valid_labels_path, name)) == 0])

    test_labels_count = len(os.listdir(test_labels_path))
    test_empty_labels_count = len([name for name in os.listdir(
        test_labels_path) if os.path.getsize(os.path.join(test_labels_path, name)) == 0])

    # Calculating the total labelled images and total unlabelled images
    total_labelled = train_labels_count + valid_labels_count + test_labels_count - \
        train_empty_labels_count - valid_empty_labels_count - test_empty_labels_count
    total_unlabelled = train_empty_labels_count + \
        valid_empty_labels_count + test_empty_labels_count

    print(
        f"Train - Labelled: {train_labels_count - train_empty_labels_count}, Unlabelled: {train_empty_labels_count}")
    print(
        f"Validation - Labelled: {valid_labels_count - valid_empty_labels_count}, Unlabelled: {valid_empty_labels_count}")
    print(
        f"Test - Labelled: {test_labels_count - test_empty_labels_count}, Unlabelled: {test_empty_labels_count}")

    # Stacked bar graph
    labels = ['Train', 'Validation', 'Test']
    labelled_counts = [train_labels_count - train_empty_labels_count, valid_labels_count -
                       valid_empty_labels_count, test_labels_count - test_empty_labels_count]
    unlabelled_counts = [train_empty_labels_count,
                         valid_empty_labels_count, test_empty_labels_count]

    width = 0.5  # the width of the bars
    fig, ax = plt.subplots()

    ax.bar(labels, labelled_counts, width, label='Labelled')
    ax.bar(labels, unlabelled_counts, width,
           label='Background', bottom=labelled_counts)

    ax.set_ylabel('Images')
    ax.set_title('Background Image Distribution')
    ax.legend()

    plt.show()


def plot_results(runs_directory, hyp_files):
    """
    Plots the highest values for a single column from results.csv for each run in the experiment.
    """

    best_values = []

    # Iterate through each file and grab highest value
    for file in hyp_files:

        results_path = os.path.join(
            "yolov5", runs_directory, file, "results.csv")
        # Check for file
        if os.path.exists(results_path):
            # Read the CSV into a DataFrame
            df = pd.read_csv(results_path)

            # Check if the results_column exists in the dataframe
            if results_column in df.columns:
                # Grab highest value from results_column
                best_map = df[results_column].max()
                best_values.append(best_map)
            else:
                print(
                    f"Warning: Column '{results_column}' not found in {results_path}.")
                best_values.append(0)
        else:
            print(f"Warning: {results_path} not found.")
            best_values.append(0)

    print(best_values)

    plt.figure(figsize=(10, 5))
    plt.bar(hyp_files, best_values, color='blue')
    plt.ylim(0.85, 0.87)
    plt.ylabel(f"Best {results_column} Over {train_epochs} Epochs")
    plt.xlabel('Runs')
    plt.title(f"Best {results_column} Values Across {len(hyp_files)} Runs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
