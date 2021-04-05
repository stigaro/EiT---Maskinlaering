import cv2
import torch
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_curve, roc_auc_score
from torchvision.ops import nms
from torchvision.transforms import functional

from sources.utility import load_json_file


class Visualizer:

    def __init__(self, outputs, targets, images, sq_num):
        """
            Constructor for visualisation class.
            args:
                outputs:tensor - predictions from model
                targets:tensor - data labels
                images:tensor - images to be visualised
                sq_num:int - how many pictures to plot, must be a square number

        """

        self.outputs = [np.argmax(output) for output in outputs]
        self.targets = targets
        self.images = images
        self.sq_num = sq_num

        self.pic_size = sqrt(self.sq_num)

        plt.figure(figsize=(10, 10))

        for i in range(sq_num):
            ax = plt.subplot(self.pic_size, self.pic_size, i + 1)
            plt.imshow(images[i])
            plt.title("T: " + str(self.targets[i]) + " O: " + str(self.outputs[i]))
            plt.axis("off")

        plt.show()

    def plot_ROC(self):
        """
            Plots ROC curve based on targets and outputs.
        """

        fpr, tpr, thresholds = roc_curve(self.targets, self.outputs)
        plt.plot(fpr, tpr)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('FP Rate')
        plt.ylabel('TP Rate')
        plt.show()

    def get_AUC(self):
        """
            Returns AUC score based on targets and outputs.
        """
        return roc_auc_score(self.targets, self.outputs)

    @staticmethod
    def visualize_instance_model_output(image, prediction, prediction_threshold=0.75, iou_threshold=0.50, font_size=16, merging_alpha=0.40):
        # Loads the data information for visualization
        information_dictionary = load_json_file('resources/unprocessed_data/information.json')
        category_mapping_dictionary = {
            int(category_dictionary['id']): category_dictionary['name']
            for category_dictionary
            in information_dictionary['categories']
        }

        # Create commong variables used in the instance loop
        visualized_image = image.copy()
        font = ImageFont.truetype('arial.ttf', size=font_size)
        labels = [category_mapping_dictionary[category.tolist()] for category in prediction['labels']]

        # Retrieve the indices that are above the prediction threshold and kept from non-maximum suppression
        prediction_threshold_kept_indices = [index for index, score in enumerate(prediction['scores'].tolist()) if score >= prediction_threshold]
        nms_kept_indices = nms(prediction['boxes'], prediction['scores'], iou_threshold).tolist()
        used_indices = [index for index in range(len(prediction['scores'])) if index in nms_kept_indices and index in prediction_threshold_kept_indices]

        # Because the way we apply the color masks, we instead apply all at once to avoid issues with opacity
        total_mask = np.zeros(visualized_image.size[::-1])
        for indice in used_indices:
            total_mask += np.array(functional.to_pil_image(prediction['masks'][indice], 'L'))

        # Apply the total instance mask
        color_mask = np.array(image)
        color_mask[total_mask > 0] = (255, 255, 0)
        colored_mask_image = cv2.addWeighted(color_mask, merging_alpha, np.array(visualized_image), 1 - merging_alpha, 0, np.array(visualized_image))
        visualized_image = Image.fromarray(colored_mask_image)

        # Draw the texts and rectangles if they are kept
        for indice in used_indices:
            box = prediction['boxes'][indice].tolist()
            draw = ImageDraw.Draw(visualized_image)
            draw.text((box[0], box[1] - font_size * 1.25), labels[indice], fill='white', font=font, stroke_width=2, stroke_fill='black')
            draw.rectangle(box, width=1, outline='red')

        return visualized_image



class VisualizerBinary:
    def __init__(self, preds, target_dataset,
                 n_fig_width = 5, n_fig_height = "same_as_height"):
        """
            Constructor for visualisation class.
            :param
                preds:tensor - predictions from model
                target_dataset - dataset of predicted images and true labels
                n_fig_width:int - number of figures for each row
                n_fig_height:int - number of figures for each column
        """
        self.preds = preds
        self.preds_argmax = [np.argmax(pred) for pred in preds]
        self.target_dataset = target_dataset
        self.n_fig_width = n_fig_width
        if n_fig_width == "same_as_height":
            self.n_fig_height = n_fig_width
        else:
            self.n_fig_height = n_fig_height
        self.n_fig = self.n_fig_height * self.n_fig_width

    def plot_images_and_pred(self, n_fig_width = None,
                             n_fig_height = None):
        """
            Plots images with predicted label along with true label.
            :param
                n_fig_width:int - number of figures for each row
                n_fig_height:int - number of figures for each column
        """
        if n_fig_width != None:
            self.n_fig_width = n_fig_width
        if n_fig_height != None:
            self.n_fig_height = n_fig_height
        self.n_fig = self.n_fig_height * self.n_fig_width

        plt.figure(figsize=(2*n_fig_width, 2*n_fig_height))

        k = 0
        print(self.preds_argmax)
        print(self.preds_argmax[0])
        print(self.preds_argmax[0].item())
        for (image_batch, label_batch) in self.target_dataset:
            for (image, label) in zip(image_batch, label_batch):
                plt.subplot(self.n_fig_width, self.n_fig_height, k+1)
                plt.imshow(image)
                plt.title("True: " + str(int(label)) + "\n Pred: " + str(self.preds_argmax[k]))
                plt.axis("off")
                k += 1
                if (k >= self.n_fig):
                    break
            if (k >= self.n_fig):
                break

        plt.show()

    def plot_ROC(self):
        """
            Plots ROC curve based on targets and outputs.
        """

        fpr, tpr, thresholds = roc_curve(self.targets, self.outputs)
        plt.plot(fpr, tpr)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('FP Rate')
        plt.ylabel('TP Rate')
        plt.show()

    def get_AUC(self):
        """
            Returns AUC score based on targets and outputs.
        """
        return roc_auc_score(self.targets, self.outputs)

    @staticmethod
    def visualize_instance_model_output(image, prediction, prediction_threshold=0.75, iou_threshold=0.50, font_size=16, merging_alpha=0.40):
        # Loads the data information for visualization
        information_dictionary = load_json_file('resources/unprocessed_data/information.json')
        category_mapping_dictionary = {
            int(category_dictionary['id']): category_dictionary['name']
            for category_dictionary
            in information_dictionary['categories']
        }

        # Create commong variables used in the instance loop
        visualized_image = image.copy()
        font = ImageFont.truetype('arial.ttf', size=font_size)
        labels = [category_mapping_dictionary[category.tolist()] for category in prediction['labels']]

        # Retrieve the indices that are above the prediction threshold and kept from non-maximum suppression
        prediction_threshold_kept_indices = [index for index, score in enumerate(prediction['scores'].tolist()) if score >= prediction_threshold]
        nms_kept_indices = nms(prediction['boxes'], prediction['scores'], iou_threshold).tolist()
        used_indices = [index for index in range(len(prediction['scores'])) if index in nms_kept_indices and index in prediction_threshold_kept_indices]

        # Because the way we apply the color masks, we instead apply all at once to avoid issues with opacity
        total_mask = np.zeros(visualized_image.size[::-1])
        for indice in used_indices:
            total_mask += np.array(functional.to_pil_image(prediction['masks'][indice], 'L'))

        # Apply the total instance mask
        color_mask = np.array(image)
        color_mask[total_mask > 0] = (255, 255, 0)
        colored_mask_image = cv2.addWeighted(color_mask, merging_alpha, np.array(visualized_image), 1 - merging_alpha, 0, np.array(visualized_image))
        visualized_image = Image.fromarray(colored_mask_image)

        # Draw the texts and rectangles if they are kept
        for indice in used_indices:
            box = prediction['boxes'][indice].tolist()
            draw = ImageDraw.Draw(visualized_image)
            draw.text((box[0], box[1] - font_size * 1.25), labels[indice], fill='white', font=font, stroke_width=2, stroke_fill='black')
            draw.rectangle(box, width=1, outline='red')

        return visualized_image
