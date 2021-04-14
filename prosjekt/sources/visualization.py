import cv2
import torch
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_curve, roc_auc_score
from torchvision.ops import nms
from torchvision.transforms import functional
import os

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
    """
    Class for vizualization of binary cnn model output.
    """
    def __init__(self, preds, target_images, true_labels,
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
        self.target_images = target_images
        self.true_labels = true_labels
        self.n_fig_width = n_fig_width
        if n_fig_height == "same_as_width":
            self.n_fig_height = self.n_fig_width
        else:
            self.n_fig_height = n_fig_height
        self.n_fig = self.n_fig_height * self.n_fig_width

        # initializing variables for ROC curve
        self.pred_np = np.array(self.preds)
        self.y_score = self.pred_np[:, 1]
        self.y_true = self.true_labels

    def plot_images_and_pred(self, n_fig_width = "same as initialized",
                             n_fig_height = "same as initialized",
                             save_path = "", filename = "plot_images_and_plot"):
        """
            Plots images with predicted label along with true label.
            :param
                n_fig_width:int - number of figures for each row
                n_fig_height:int - number of figures for each column
        """
        if n_fig_width != "same as initialized":
            self.n_fig_width = n_fig_width
        if n_fig_height != "same as initialized":
            self.n_fig_height = n_fig_height


        self.n_fig = self.n_fig_height * self.n_fig_width

        plt.figure(figsize=(2*self.n_fig_height, 2*self.n_fig_width))
        k = 0
        for image, label in zip(self.target_images, self.true_labels):
                ax = plt.subplot(self.n_fig_width, self.n_fig_height, k+1)
                plt.imshow(image)
                label_class = int(label)
                if label_class == self.preds_argmax[k]:
                    plt.setp(ax.spines.values(), linewidth = 4, color="green")
                else:
                    plt.setp(ax.spines.values(), linewidth = 4, color="red")
                plt.title("True: " + str(label_class) +
                          "\n Pred: " + str(self.preds_argmax[k]) +
                          " [" + str(round(float(self.preds[k][self.preds_argmax[k]]),2)) + "]")
                #plt.axis("off")
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                k += 1
                if (k >= self.n_fig):
                    break

        plt.tight_layout(pad=1.5)
        if save_path:
            if os.path.isfile(os.path.join(save_path, filename)):
                os.remove(os.path.join(save_path, filename))
            plt.savefig(os.path.join(save_path, filename))
        plt.show()

    def plot_ROC(self, save_path = "", filename = "roc_plot.jpg"):
        """
            Plots ROC curve based on predictions and true labels.
        """
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_score)
        plt.plot(fpr, tpr, label = "Model")
        x = np.linspace(0,1, 1000)
        plt.plot(x, x, label="Random chance model")
        plt.legend()
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC value: " + str(round(self.get_AUC(),4)))
        plt.suptitle("AUC plot", fontsize=16)
        if save_path:
            if os.path.isfile(os.path.join(save_path, filename)):
                os.remove(os.path.join(save_path, filename))
            plt.savefig(os.path.join(save_path, filename))
        plt.show()

        ys = np.linspace(0,1,10)
        print(np.mean(self.preds_argmax == self.y_true))

    def get_AUC(self):
        """
            Returns AUC score based on targets and outputs.
        """
        return roc_auc_score(self.y_true, self.y_score)


def plot_metrics(history, save_path = "", filename = "plot_metrics"):
    """
        Plots history metrics.
        args:
            history:dict - models output history
    """

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    if save_path:
        if os.path.isfile(os.path.join(save_path, filename)):
            os.remove(os.path.join(save_path, filename))
        plt.savefig(os.path.join(save_path, filename))
    plt.show()