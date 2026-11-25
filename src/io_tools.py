# External Libs
import torchvision.transforms as transforms

# Transform compatible with torchxrayvision models
torchxrayvision_transform = transforms.Compose([
    # Rescale
    transforms.Resize((224,224)),

    # Apply grayscale
    transforms.Grayscale(num_output_channels=1),

    # Convert to tensor
    transforms.ToTensor(),

    # normalize to range [-1024,1024]
    transforms.Lambda(lambda x: x * 2048 - 1024)
])