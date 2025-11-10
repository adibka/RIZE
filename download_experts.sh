#!/bin/bash
# download_experts.sh - places .pkl files directly in RIZE/experts/

mkdir -p experts
echo "Downloading experts.zip..."
gdown "https://drive.google.com/uc?id=1wWMOaIl-qhXW_7mB0laXwAj5Cnf6Tmv5" -O experts/experts.zip

unzip -q experts/experts.zip -d temp_experts  # extract to temp
mv temp_experts/experts/* experts/           # move contents up
rm -rf temp_experts experts/experts.zip      # clean up

echo "Done. Expert demos in: $(pwd)/experts/"
