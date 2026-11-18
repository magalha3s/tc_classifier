import torchvision.transforms as transforms

# default_transform
default_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])