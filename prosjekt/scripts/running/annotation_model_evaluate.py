import torch

from sources.dataset import TacoDataset
from libraries.vision_master.references.detection import engine, utils
from sources import transformation
from sources.model import get_model_instance_segmentation


__MODEL_PATH = './resources/models'


def get_transform():
    return transformation.Compose([
        transformation.Resize((1024, 1024)),
        transformation.ToTensor()
    ])


if __name__ == '__main__':
    # Use the GPU or CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Retrieve the testing dataset
    testing_dataset = TacoDataset(folder_path='resources/datasets/trash_annotation_dataset/testing', transforms=get_transform())
    data_loader_testing = torch.utils.data.DataLoader(testing_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

    # Generate a pre-trained model
    number_of_classes = len(testing_dataset.information['categories']) + 1  # We add one because the background is a implicit class already
    model = get_model_instance_segmentation(number_of_classes)
    model.load_state_dict(torch.load(__MODEL_PATH + '/' + 'model09042021_dict', map_location=torch.device('cuda')), strict=False)
    model.to(device)
    model.eval()

    print("Evaluating model...")
    # Evaluate on the test datasets
    engine.evaluate(model, data_loader_testing, device=device)
    print("Finished!")
