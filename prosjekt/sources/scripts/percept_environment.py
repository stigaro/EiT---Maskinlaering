import torch
from PIL import Image
from torchvision.transforms import functional

from sources.visualization import Visualizer
from sources.model import get_model_instance_segmentation

__ENVIRONMENT_PATH = './resources/perception_environment'
__MODEL_PATH = './resources/models'

model = get_model_instance_segmentation(61)
model.load_state_dict(torch.load(__MODEL_PATH + '/' + input('Enter name of model file to load'), map_location=torch.device('cpu')), strict=False)
model.eval()

predict = lambda image: model([functional.to_tensor(image)])[0]
load_image = lambda image_path: Image.open(image_path).convert('RGB')

image_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
for image_name in image_names:
    image = load_image(__ENVIRONMENT_PATH + '/' + image_name)
    prediction = predict(image)
    visualized_image = Visualizer.visualize_instance_model_output(image, prediction, iou_threshold=0.25)
    visualized_image.show()
    visualized_image.save(__ENVIRONMENT_PATH + '/' + '[PREDICTION]_' + image_name)
