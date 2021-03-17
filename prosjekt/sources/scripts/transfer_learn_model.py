import torch

from sources.dataset import TacoDataset
from libraries.vision_master.references.detection import engine, utils
from sources import transformation
from sources.model import get_model_instance_segmentation


def get_transform():
    return transformation.Compose([
        transformation.Resize((1024, 1024)),
        transformation.ToTensor()
    ])


if __name__ == '__main__':
    # Train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Retrieve the training and testing dataset
    training_dataset = TacoDataset(folder_path='resources/processed_data/training', transforms=get_transform())
    testing_dataset = TacoDataset(folder_path='resources/processed_data/testing', transforms=get_transform())

    # Define training and testing data loaders
    data_loader_training = torch.utils.data.DataLoader(training_dataset, batch_size=2, shuffle=True, num_workers=1,
                                                       collate_fn=utils.collate_fn)
    data_loader_testing = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                      collate_fn=utils.collate_fn)

    # Set the number of classes
    number_of_classes = len(training_dataset.information['categories']) + 1  # We add one because the background is a implicit class already

    # Generate a pre-trained model
    model = get_model_instance_segmentation(number_of_classes)

    # Move the model to the right device
    model.to(device)

    # Construct a optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Construct a Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Train for 10 epochs
    number_of_epochs = 5
    for epoch in range(number_of_epochs):
        # Train for one epoch, printing every 10 iterations
        engine.train_one_epoch(model, optimizer, data_loader_training, device, epoch, print_freq=1)

        # Update the learning rate
        lr_scheduler.step()

        # Evaluate on the test dataset
        # TODO: engine.evaluate(model, data_loader_testing, device=device)
