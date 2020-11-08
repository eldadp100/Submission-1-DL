import os
import torch
from models import CommandRecognitionNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoints_folder = os.path.join('.', 'checkpoints')
net = CommandRecognitionNet()

# load checkpoint:
checkpoint_path = os.path.join(checkpoints_folder, 'net_checkpoint.pt')
loaded_state_dict = torch.load(checkpoint_path, map_location=device)
net.load_state_dict(loaded_state_dict)

# play with the loaded model...
