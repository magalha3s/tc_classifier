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
if [[ "$(cat /etc/hostname)" == *"bancada"* || "$(cat /etc/hostname)" == *"caloba"* ]]; then
    virtualenv -p python .venv
else
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install --progress-bar off --upgrade pip
pip install --progress-bar off -r requirements.txt

echo -e "\033[32mEnvironment setup completed successfully!\033[0m"