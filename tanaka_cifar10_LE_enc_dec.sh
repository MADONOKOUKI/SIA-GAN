cd /groups1/gaa50073/madono/aaai_ws_code_autumun
source ~/.bashrc
module load python/3.6/3.6.5
python3 -m venv libraly_experiments
source libraly_experiments/bin/activate
module load python/3.6/3.6.5
module load cuda/10.1/10.1.243
module load cudnn/7.4/7.4.2
module load nccl/2.4/2.4.7-1
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
apt-get install python-dev
python3 -m pip install matplotlib
pip3 install torch torchvision tensorboardX
pip install -U scikit-learn
pip freeze > requirements.txt
rm -rf tanaka_adaptation_network_cifar10_LE
rm -rf tanaka_adaptation_network_cifar10_LE.t7
rm -rf tanaka_adaptation_network_cifar10_LE.json 
val=1
mask=False
python tanaka_cifar10_LE_enc_dec.py --num_of_keys="$val" --e=155 --mask "$mask" --tensorboard_name tanaka_adaptation_network_cifar10_LE_repeat_"$mask" --training_model_name tanaka_adaptation_network_cifar10_LE_repeat_"$mask".t7 --json_file_name tanaka_adaptation_network_cifar10_LE_repeat_"$mask".json > tanaka_cifar10_LE_enc_dec_"$mask".txt
