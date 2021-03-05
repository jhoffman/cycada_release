export PYTHONPATH=".:${PYTHONPATH} "

# Setting up necessary python paths to link the modules of upsnet and cycada works
export PYTHONPATH="/home/vkonduru/domain/cycada/UPSNet/:/home/vkonduru/domain/cycada/:${PYTHONPATH}"

# Dependencies
# CYCADA
pip install -r requirements.txt

# UPSNET
cd UPSNet
./init.sh