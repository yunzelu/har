# Import necessary libraries
import kagglehub
import shutil

# Download the dataset from Kaggle using its identifier

# path = kagglehub.dataset_download("ngoduy/dataset-video-for-human-action-recognition")
path = kagglehub.dataset_download("sharjeelmazhar/human-activity-recognition-video-dataset")

# Print the path where the dataset files are downloaded
print("Path to dataset files:", path)

# Move the downloaded dataset to the desired directory
# shutil.move(path, "/Users/yasas/Documents/coursework-comp5013-panicatthekernel")
