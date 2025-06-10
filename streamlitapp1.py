# Import the NumPy library for numerical computing

import numpy as np



# Import the PIL library for image manipulation

from PIL import Image



# Import the PyTorch library for deep learning

import torch



# Import PyTorch's data loading utilities

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler



# Import PyTorch's image transformation utilities

from torchvision import transforms



# Import scikit-learn's utilities for splitting data into train/test sets and for k-fold cross-validation

from sklearn.model_selection import train_test_split, StratifiedKFold



# Import the Matplotlib library for plotting

import matplotlib.pyplot as plt



# Import Matplotlib's Rectangle object for drawing bounding boxes

from matplotlib.patches import Rectangle



# Import the Seaborn library for visualization

import seaborn as sns



# Import scikit-learn's utilities for evaluating classification performance

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score



# Define a constant string variable for the name of the folder containing the "good" class images

GOOD_CLASS_FOLDER = "good"



# Define a constant list variable for the dataset splits

DATASET_SETS = ["train", "test"]



# Define a constant string variable for the format of the image files

IMG_FORMAT = ".png"



# Define a constant tuple variable for the size of the input images

INPUT_IMG_SIZE = (224, 224)



# Define a constant integer variable for the negative class label

NEG_CLASS = 1



# Import the warnings module to suppress warnings in the code

import warnings



# Ignore all warnings that may be raised during the code execution

warnings.filterwarnings("ignore")





class MVTEC_AD_DATASET(Dataset):

    """

    Class to load subsets of MVTEC ANOMALY DETECTION DATASET

    Dataset Link: https://www.mvtec.com/company/research/datasets/mvtec-ad

   

    Root is path to the subset, for instance, `mvtec_anomaly_detection/leather`

    """



    def __init__(self, root):

        # Define the class labels based on the NEG_CLASS setting.

        self.classes = ["Good", "Anomaly"] if NEG_CLASS == 1 else ["Anomaly", "Good"]

        # Define the image transformation pipeline.

        self.img_transform = transforms.Compose(

            [transforms.Resize(INPUT_IMG_SIZE), transforms.ToTensor()]

        )



        # Load the image filenames and labels for the dataset.

        (

            self.img_filenames,

            self.img_labels,

            self.img_labels_detailed,

        ) = self._get_images_and_labels(root)



    def _get_images_and_labels(self, root):

        # Initialize lists to store image filenames and labels.

        image_names = []

        labels = []

        labels_detailed = []



        # Loop over the dataset sets (e.g., "train", "test") and classes ("good" and "anomaly").

        for folder in DATASET_SETS:

            # Construct the path to the class folder.

            folder = os.path.join(root, folder)



            # Loop over the class folders in the dataset.

            for class_folder in os.listdir(folder):

                # Determine the label for the class based on its folder name.

                label = (

                    1 - NEG_CLASS if class_folder == GOOD_CLASS_FOLDER else NEG_CLASS

                )

                # Store the detailed label (i.e., the class folder name).

                label_detailed = class_folder



                # Construct the path to the class image folder.

                class_folder = os.path.join(folder, class_folder)

                # Get the list of image filenames in the class folder that match the IMG_FORMAT setting.

                class_images = os.listdir(class_folder)

                class_images = [

                    os.path.join(class_folder, image)

                    for image in class_images

                    if image.find(IMG_FORMAT) > -1

                ]



                # Add the class image filenames and labels to the respective lists.

                image_names.extend(class_images)

                labels.extend([label] * len(class_images))

                labels_detailed.extend([label_detailed] * len(class_images))



        # Print some statistics about the dataset.

        print(

            "Dataset {}: N Images = {}, Share of anomalies = {:.3f}".format(

                root, len(labels), np.sum(labels) / len(labels)

            )

        )

        # Return the lists of image filenames and labels.

        return image_names, labels, labels_detailed



    def __len__(self):

        # Return the length of the dataset (i.e., the number of images).

        return len(self.img_labels)



    def __getitem__(self, idx):

        # Get the filename and label for the image at the specified index.

        img_fn = self.img_filenames[idx]

        label = self.img_labels[idx]

        # Open the image file and apply the image transformation pipeline.

        img = Image.open(img_fn)

        img = self.img_transform(img)

        # Convert the label to a PyTorch tensor.

        label = torch.as_tensor(label, dtype=torch.long)

        # Return the transformed image and label as a tuple.

        return img, label

   

    # This function takes in the root directory of the MVTEC_AD dataset, batch size for DataLoader, test_size, and random_state as input arguments.

def get_train_test_loaders(root, batch_size, test_size=0.2, random_state=42):

    """

    Returns train and test dataloaders.

    Splits dataset in stratified manner, considering various defect types.

    """

    # Initialize the dataset object with the given root directory.

    dataset = MVTEC_AD_DATASET(root=root)



    # Split the indices of dataset into train and test sets in a stratified manner based on the defect types.

    train_idx, test_idx = train_test_split(

        np.arange(dataset.__len__()),

        test_size=test_size,

        shuffle=True,

        stratify=dataset.img_labels_detailed,

        random_state=random_state,

    )



    # Initialize the SubsetRandomSampler for the training set and test set.

    train_sampler = SubsetRandomSampler(train_idx)

    test_sampler = SubsetRandomSampler(test_idx)



    # Initialize the DataLoader objects for the training set and test set with the SubsetRandomSampler.

    train_loader = DataLoader(

        dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True

    )

    test_loader = DataLoader(

        dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False

    )



    # Return the DataLoader objects for the training set and test set.

    return train_loader, test_loader





# This function takes in the root directory of the MVTEC_AD dataset, batch size for DataLoader, and n_folds as input arguments.

def get_cv_train_test_loaders(root, batch_size, n_folds=5):

    """

    Returns train and test dataloaders for N-Fold cross-validation.

    Splits dataset in stratified manner, considering various defect types.

    """

    # Initialize the dataset object with the given root directory.

    dataset = MVTEC_AD_DATASET(root=root)



    # Initialize the StratifiedKFold object for the specified number of folds.

    kf = StratifiedKFold(n_splits=n_folds)



    # Initialize an empty list for storing the DataLoader objects for each fold.

    kf_loader = []



    # Split the dataset into train and test sets for each fold using the StratifiedKFold object.

    for train_idx, test_idx in kf.split(

        np.arange(dataset.__len__()), dataset.img_labels_detailed

    ):

        # Initialize the SubsetRandomSampler for the training set and test set.

        train_sampler = SubsetRandomSampler(train_idx)

        test_sampler = SubsetRandomSampler(test_idx)



        # Initialize the DataLoader objects for the training set and test set with the SubsetRandomSampler.

        train_loader = DataLoader(

            dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True

        )

        test_loader = DataLoader(

            dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False

        )



        # Append the DataLoader objects for the current fold to the list.

        kf_loader.append((train_loader, test_loader))



    # Return the list of DataLoader objects for all folds.

    return kf_loader

# This function takes in the root directory of the MVTEC_AD dataset, batch size for DataLoader, test_size, and random_state as input arguments.

def get_train_test_loaders(root, batch_size, test_size=0.2, random_state=42):

    """

    Returns train and test dataloaders.

    Splits dataset in stratified manner, considering various defect types.

    """

    # Initialize the dataset object with the given root directory.

    dataset = MVTEC_AD_DATASET(root=root)



    # Split the indices of dataset into train and test sets in a stratified manner based on the defect types.

    train_idx, test_idx = train_test_split(

        np.arange(dataset.__len__()),

        test_size=test_size,

        shuffle=True,

        stratify=dataset.img_labels_detailed,

        random_state=random_state,

    )



    # Initialize the SubsetRandomSampler for the training set and test set.

    train_sampler = SubsetRandomSampler(train_idx)

    test_sampler = SubsetRandomSampler(test_idx)



    # Initialize the DataLoader objects for the training set and test set with the SubsetRandomSampler.

    train_loader = DataLoader(

        dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True

    )

    test_loader = DataLoader(

        dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False

    )



    # Return the DataLoader objects for the training set and test set.

    return train_loader, test_loader





# This function takes in the root directory of the MVTEC_AD dataset, batch size for DataLoader, and n_folds as input arguments.

def get_cv_train_test_loaders(root, batch_size, n_folds=5):

    """

    Returns train and test dataloaders for N-Fold cross-validation.

    Splits dataset in stratified manner, considering various defect types.

    """

    # Initialize the dataset object with the given root directory.

    dataset = MVTEC_AD_DATASET(root=root)



    # Initialize the StratifiedKFold object for the specified number of folds.

    kf = StratifiedKFold(n_splits=n_folds)



    # Initialize an empty list for storing the DataLoader objects for each fold.

    kf_loader = []



    # Split the dataset into train and test sets for each fold using the StratifiedKFold object.

    for train_idx, test_idx in kf.split(

        np.arange(dataset.__len__()), dataset.img_labels_detailed

    ):

        # Initialize the SubsetRandomSampler for the training set and test set.

        train_sampler = SubsetRandomSampler(train_idx)

        test_sampler = SubsetRandomSampler(test_idx)



        # Initialize the DataLoader objects for the training set and test set with the SubsetRandomSampler.

        train_loader = DataLoader(

            dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True

        )

        test_loader = DataLoader(

            dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False

        )



        # Append the DataLoader objects for the current fold to the list.

        kf_loader.append((train_loader, test_loader))



    # Return the list of DataLoader objects for all folds.

    return kf_loader



# This function takes in the dataloader (for loading training data), the model, optimizer, loss criterion, number of epochs, device to train the model on (CPU or GPU), and an optional target accuracy to stop training early.

def train(

    dataloader, model, optimizer, criterion, epochs, device, target_accuracy=None

):

    """

    Script to train a model. Returns trained model.

    """



    # These lines move the model to the specified device (CPU or GPU) and puts the model in train mode.

    model.to(device)

    model.train()



    # This loop iterates over the number of epochs specified and initializes variables to track loss, accuracy, and number of samples processed during training

    for epoch in range(1, epochs + 1):

        print(f"Epoch {epoch}/{epochs}:", end=" ")

        running_loss = 0

        running_corrects = 0

        n_samples = 0



        # This inner loop iterates over batches of data from the dataloader, moves the inputs and labels to the specified device, performs forward and backward passes through the model, calculates the loss and updates the model weights via the optimizer, and updates the running loss and accuracy statistics.

        for inputs, labels in dataloader:

            inputs = inputs.to(device)

            labels = labels.to(device)



            optimizer.zero_grad()

            preds_scores = model(inputs)

            preds_class = torch.argmax(preds_scores, dim=-1)

            loss = criterion(preds_scores, labels)

            loss.backward()

            optimizer.step()



            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds_class == labels)

            n_samples += inputs.size(0)



       

        # This code calculates the average loss and accuracy over the epoch and prints the results. If a target accuracy is specified, the code checks if the current epoch's accuracy exceeds the target and stops training early if it does.

        epoch_loss = running_loss / n_samples

        epoch_acc = running_corrects.double() / n_samples

        print("Loss = {:.4f}, Accuracy = {:.4f}".format(epoch_loss, epoch_acc))



        if target_accuracy != None:

            if epoch_acc > target_accuracy:

                print("Early Stopping")

                break



    # This function returns the trained model.

    return model



import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import models

import torch.optim as optim



# Set input image size

INPUT_IMG_SIZE = (224, 224)



class CustomVGG(nn.Module):

    """

    Custom multi-class classification model

    with VGG16 feature extractor, pretrained on ImageNet

    and custom classification head.

    Parameters for the first convolutional blocks are freezed.

   

    Returns class scores when in train mode.

    Returns class probs and normalized feature maps when in eval mode.

    """



    def __init__(self, n_classes=2):

        super(CustomVGG, self).__init__()



        # Load VGG16 feature extractor, pretrained on ImageNet

        self.feature_extractor = models.vgg16(pretrained=True).features[:-1]



        # Define custom classification head

        self.classification_head = nn.Sequential(

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AvgPool2d(

                kernel_size=(INPUT_IMG_SIZE[0] // 2 ** 5, INPUT_IMG_SIZE[1] // 2 ** 5)

            ),

            nn.Flatten(),

            nn.Linear(

                in_features=self.feature_extractor[-2].out_channels,

                out_features=n_classes,

            ),

        )



        # Freeze parameters for the first convolutional blocks of the feature extractor

        self._freeze_params()



    def _freeze_params(self):

        # Loop through all parameters for the first 23 convolutional blocks

        for param in self.feature_extractor[:23].parameters():

            # Freeze parameters

            param.requires_grad = False



    def forward(self, x):

        # Compute feature maps using VGG16 feature extractor

        feature_maps = self.feature_extractor(x)



        # Compute class scores using custom classification head

        scores = self.classification_head(feature_maps)



        # If in training mode, return class scores

        if self.training:

            return scores



        # If in evaluation mode, return class probabilities and normalized feature maps

        else:

            # Compute class probabilities from class scores using softmax activation function

            probs = nn.functional.softmax(scores, dim=-1)



            # Compute normalized feature maps from classification head weights and feature maps

            weights = self.classification_head[3].weight

            weights = (

                weights.unsqueeze(-1)

                .unsqueeze(-1)

                .unsqueeze(0)

                .repeat(

                    (

                        x.size(0),

                        1,

                        1,

                        INPUT_IMG_SIZE[0] // 2 ** 4,

                        INPUT_IMG_SIZE[0] // 2 ** 4,

                    )

                )

            )

            feature_maps = feature_maps.unsqueeze(1).repeat((1, probs.size(1), 1, 1, 1))

            location = torch.mul(weights, feature_maps).sum(axis=2)

            location = F.interpolate(location, size=INPUT_IMG_SIZE, mode="bilinear")



            # Normalize feature maps to range [0, 1]

            maxs, _ = location.max(dim=-1, keepdim=True)

            maxs, _ = maxs.max(dim=-2, keepdim=True)

            mins, _ = location.min(dim=-1, keepdim=True)

            mins, _ = mins.min(dim=-2, keepdim=True)

            norm_location = (location - mins) / (maxs - mins)



            # Return class probabilities and normalized feature maps

            return probs, norm_location

   

    def evaluate(model, dataloader, device):

        # Move the model to the specified device

        model.to(device)

        # Set the model to evaluation mode

        model.eval()

        # Get the class names from the dataloader's dataset

        class_names = dataloader.dataset.classes

   

        # Initialize variables to keep track of correct predictions, true labels, and predicted labels

        running_corrects = 0

        y_true = np.empty(shape=(0,))

        y_pred = np.empty(shape=(0,))

   

        # Loop through the dataloader's batches

        for inputs, labels in dataloader:

            # Move the inputs and labels to the specified device

            inputs = inputs.to(device)

            labels = labels.to(device)

   

            # Forward pass the inputs through the model and get the predicted probabilities and classes

            preds_probs = model(inputs)[0]

            preds_class = torch.argmax(preds_probs, dim=-1)

   

            # Move the labels and predicted classes to the CPU and convert them to numpy arrays

            labels = labels.to("cpu").numpy()

            preds_class = preds_class.detach().to("cpu").numpy()

   

            # Concatenate the true labels and predicted labels to the respective arrays

            y_true = np.concatenate((y_true, labels))

            y_pred = np.concatenate((y_pred, preds_class))

   

        # Calculate the accuracy and balanced accuracy scores using scikit-learn's metrics

        accuracy = accuracy_score(y_true, y_pred)

        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

   

        # Print the accuracy and balanced accuracy scores

        print("Accuracy: {:.4f}".format(accuracy))

        print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))

        print()

        # Plot the confusion matrix using seaborn's heatmap function

        plot_confusion_matrix(y_true, y_pred, class_names=class_names)





def plot_confusion_matrix(y_true, y_pred, class_names="auto"):

    # Calculate the confusion matrix using scikit-learn's metrics

    confusion = confusion_matrix(y_true, y_pred)

    # Create a new figure with a specified size

    plt.figure(figsize=[5, 5])

    # Plot the confusion matrix as a heatmap using seaborn's heatmap function

    sns.heatmap(

        confusion,

        annot=True,

        cbar=False,

        xticklabels=class_names,

        yticklabels=class_names,

    )



    # Set the labels and title of the plot

    plt.ylabel("True labels")

    plt.xlabel("Predicted labels")

    plt.title("Confusion Matrix")

    # Display the plot

    plt.show()

   

   

def get_bbox_from_heatmap(heatmap, thres=0.8):

    """

    Returns bounding box around the defected area:

    Upper left and lower right corner.



    Threshold affects size of the bounding box.

    The higher the threshold, the wider the bounding box.

    """

    # Create a binary map by thresholding the heatmap

    binary_map = heatmap > thres



    # Compute the x-coordinate of the left and right edge of the bounding box

    x_dim = np.max(binary_map, axis=0) * np.arange(0, binary_map.shape[1])

    x_0 = int(x_dim[x_dim > 0].min())

    x_1 = int(x_dim.max())



    # Compute the y-coordinate of the top and bottom edge of the bounding box

    y_dim = np.max(binary_map, axis=1) * np.arange(0, binary_map.shape[0])

    y_0 = int(y_dim[y_dim > 0].min())

    y_1 = int(y_dim.max())



    # Return the four corners of the bounding box

    return x_0, y_0, x_1, y_1





# The function shows the image, its true label, predicted label and predicted probability.

# If the model predicts an anomaly, the function draws a bounding box (bbox) around the defected region and a heatmap.

# The plot displays the images in a grid, with each image and its label/prediction information in one subplot.

def predict_localize(

    model, dataloader, device, thres=0.8, n_samples=9, show_heatmap=False

):

    """

    Runs predictions for the samples in the dataloader.

    Shows image, its true label, predicted label and probability.

    If an anomaly is predicted, draws bbox around defected region and heatmap.

    """



    # Move model to device and set to evaluation mode

    model.to(device)

    model.eval()



    # Get class names from dataloader

    class_names = dataloader.dataset.classes

   

    # Convert PyTorch tensor to PIL Image for displaying images

    transform_to_PIL = transforms.ToPILImage()



    # Calculate number of rows and columns for subplot visualization

    n_cols = 3

    n_rows = int(np.ceil(n_samples / n_cols))

   

    # Set figure size

    plt.figure(figsize=[n_cols * 5, n_rows * 5])



    # Initialize sample counter

    counter = 0

   

    # Iterate over batches in dataloader

    for inputs, labels in dataloader:

       

        # Move batch to device

        inputs = inputs.to(device)

       

        # Generate predictions and feature maps from model

        out = model(inputs)

        probs, class_preds = torch.max(out[0], dim=-1)

        feature_maps = out[1].to("cpu")



        # Iterate over images in batch

        for img_i in range(inputs.size(0)):

           

            # Get image, predicted label, probability, and true label

            img = transform_to_PIL(inputs[img_i])

            class_pred = class_preds[img_i]

            prob = probs[img_i]

            label = labels[img_i]

           

            # Get heatmap for negative class (anomaly) if predicted

            heatmap = feature_maps[img_i][NEG_CLASS].detach().numpy()



            # Increment subplot counter

            counter += 1

           

            # Create subplot for image

            plt.subplot(n_rows, n_cols, counter)

           

            # Show image and set axis off

            plt.imshow(img)

            plt.axis("off")

           

            # Set title with predicted label, probability, and true label

            plt.title(

                "Predicted: {}, Prob: {:.3f}, True Label: {}".format(

                    class_names[class_pred], prob, class_names[label]

                )

            )



            # If anomaly is predicted (negative class)

            if class_pred == NEG_CLASS:

               

                # Get bounding box from heatmap and draw rectangle around anomaly

                x_0, y_0, x_1, y_1 = get_bbox_from_heatmap(heatmap, thres)

                rectangle = Rectangle(

                    (x_0, y_0),

                    x_1 - x_0,

                    y_1 - y_0,

                    edgecolor="red",

                    facecolor="none",

                    lw=3,

                )

                plt.gca().add_patch(rectangle)

               

                # If show_heatmap is True, show heatmap

                if show_heatmap:

                    plt.imshow(heatmap, cmap="Reds", alpha=0.3)



            # If counter equals number of samples, show plot and return

            if counter == n_samples:

                plt.tight_layout()

                plt.show()

                return
