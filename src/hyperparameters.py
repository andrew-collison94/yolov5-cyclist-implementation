import os

# How many images to select for random sampling
num_samples = 5

# File paths
train_images_path = os.path.join('data', 'cyclist_data', 'images', 'train')
train_labels_path = os.path.join('data', 'cyclist_data', 'labels', 'train')
test_images_path = os.path.join('data', 'cyclist_data', 'images', 'test')
test_labels_path = os.path.join('data', 'cyclist_data', 'labels', 'test')
valid_images_path = os.path.join('data', 'cyclist_data', 'images', 'valid')
valid_labels_path = os.path.join('data', 'cyclist_data', 'labels', 'valid')


train_batch_size = 8
train_epochs = 20

valid_batch_size = 8

'''
List of yaml files to expiriment with iou values.
This is set to 0.20 by default in "hyp.scratch-low.yaml". 
The following yaml files have a are set to:
hyp_iou_1.yaml - iou_t = 0.10
hyp_iou_2.yaml - iou_t = 0.20
and so on.
'''
hyp_iou_files = ["hyp_iou_1.yaml", "hyp_iou_2.yaml", "hyp_iou_3.yaml",
                 "hyp_iou_4.yaml", "hyp_iou_5.yaml", "hyp_iou_6.yaml"]

runs_directory = 'train_cyclists'

# Named of the column from the results.csv where the best value for the run should be plotted
# All column headers in the results.csv have 5 spaces in them
results_column = '     metrics/mAP_0.5'
