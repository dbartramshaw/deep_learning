# AWS deep learning AMI fix

# changes to dataloader
'/home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/utils/data/'

# open
vim dataloader.py

#add this
try:
   FileNotFoundError
except NameError:
   FileNotFoundError = IOError

pip install tensorboardX
(pytorch_p36) ubuntu@ip-172-31-52-40:~/data$ 