. C:/Users/DELL/Anaconda3/etc/profile.d/conda.sh
conda create --prefix ./env python=3.7 -y
conda activate ./env
pip install -r requirements.txt
conda env export > conda.yaml