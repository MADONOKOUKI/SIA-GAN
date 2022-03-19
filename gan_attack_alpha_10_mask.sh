cd /groups1/gaa50073/madono/aaai_ws_code_autumun
source ~/.bashrc
module load python/3.6/3.6.5
python3 -m venv work_gan
source work_gan/bin/activate
module load python/3.6/3.6.5
module load nccl/2.2/2.2.13-1
module load python/3.6/3.6.5
module load cuda/10.1/10.1.243
module load 7.6/7.6.4
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
apt-get install python-dev
python3 -m pip install matplotlib
pip3 install torch torchvision tensorboardX
pip install -U scikit-learn
pip install opencv-python
pip freeze
val=2
mixup=True
noise=False
alpha=0.2
rm -rf repositories/gan_attack_imgs_"$val"_"$mixup"_"$alpha"_"$noise"_mask
rm -rf repositories/gan_attack_tensorboard_"$val"_"$mixup"_"$alpha"_"$noise"_mask
mkdir repositories/gan_attack_imgs_"$val"_"$mixup"_"$alpha"_"$noise"_mask
mkdir repositories/gan_attack_tensorboard_"$val"_"$mixup"_"$alpha"_"$noise"_mask
python gan_attack_alpha_10_mask.py --alpha "$alpha" --mixup "$mixup" --num_of_keys="$val" --n_epochs 100 --batch_size 256 --lr 0.0001 --b1 0.0 --b2 0.9 --n_cpu 8 --img_size 32 --channels 3 --tensorboard_name repositories/gan_attack_tensorboard_"$val"_"$mixup"_"$alpha"_"$noise"_mask --directory_name repositories/gan_attack_imgs_"$val"_"$mixup"_"$alpha"_"$noise"_mask --adversarial_loss hinge > repositories/gan_attack_reconstrcution_"$val"_"$mixup"_"$alpha"_"$noise"_mask.txt
