#!/bin/bash
echo "

 _____  ___   __      _               
/__   \/ __\ / _\ ___| |_ _   _ _ __  
  / /\/ /    \ \ / _ \ __| | | | '_ \ 
 / / / /___  _\ \  __/ |_| |_| | |_) |
 \/  \____/  \__/\___|\__|\__,_| .__/ 
                               |_|    

"

# Check if environment exists
if [[ -d ".venv" ]]; then
    printf "\033[0;31mVirtual environment already exists!\033[0m\n"
    exit 1
fi


# Create virtual environment
if [[ "$(hostname)" == *"bancada"* ]]; then
    virtualenv -p python .venv
else
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install --quiet --upgrade pip
pip install --quiet torch torchvision matplotlib numpy pandas Pillow torchxrayvision gdown

clear

echo -e "\033[32mEnvironment setup completed successfully!"
echo -e "Starting weights download!\033[0m"

# Setup file name and id
FILEID=""
FILENAME=""

# Download weights
if ! gdown "$FILEID" -O "$FILENAME" --continue --retries 5; then
    printf "\033[0;31mInvalid fileid or filename. Download failed!\033[0m\n"
    exit 1
fi