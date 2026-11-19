import torchvision.transforms as transforms

# default_transform
default_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2048 - 1024),
])