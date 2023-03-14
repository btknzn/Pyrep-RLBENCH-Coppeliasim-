# Standard imports
import numpy as np
import os

# Imports for the simulator
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from torchvision.models import resnet18
# Imports for the neural model
from qLearner import qAgent
from qLearner import training_tracker
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset


resnet = resnet50(True)
resnet_layers = list(resnet.children())[:-1]
encoderRGB = nn.Sequential(*resnet_layers)


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
def setup_env(headless=True, planning=True):
    # Set all unnecessary stuff to False to speed up simulation
    obs_config = ObservationConfig()
    obs_config.left_shoulder_camera.set_all(True)
    obs_config.right_shoulder_camera.set_all(True)
    obs_config.overhead_camera.set_all(False)
    obs_config.wrist_camera.set_all(False)
    obs_config.front_camera.set_all(True)
    obs_config.joint_velocities = True
    obs_config.joint_positions = True
    obs_config.joint_forces = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = False
    obs_config.gripper_matrix = False
    obs_config.gripper_joint_positions = False
    obs_config.gripper_touch_forces = False
    obs_config.wrist_camera_matrix = False
    obs_config.task_low_dim_state = True
    # obs_config.set_all(True)

    arm_action_mode = MoveArmThenGripper(
        arm_action_mode=JointPosition(),
        gripper_action_mode=Discrete()
    )

    env = Environment(arm_action_mode, obs_config=obs_config, headless=headless)
    env.launch()
    return env, env.get_task(ReachTarget)



class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,16,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,16,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8,kernel_size=5,stride=2),
            nn.ReLU(),
            )
    def forward(self,x):
        x = self.encoder(x)
        return x
    

class predictivelearning(nn.Module):
    def __init__(self,n_hidden=512,out_dim=7):
        super(predictivelearning, self).__init__()
        self.n_hidden = n_hidden
        # projectionRGB + projectionDepth+7
        self.lstm1 = nn.LSTMCell(507,self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden,self.n_hidden)
        self.lstm3 = nn.LSTMCell(self.n_hidden,self.n_hidden)
        self.lstm4 = nn.LSTMCell(self.n_hidden,self.n_hidden)
        self.lstm5 = nn.LSTMCell(self.n_hidden,self.n_hidden)
        
        self.projectionRGB = nn.Linear(2048,400)
        self.projectionDepth = nn.Linear(200,100)
        self.encoderdepth = encoder()
        self.encoderRGB = encoderRGB
        self.linear1 = nn.Linear(self.n_hidden,32)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(32,out_dim)

    def forward(self,whiteblack,depth,joints):
        outputs = []
        whiteblackencoded = self.encoderRGB(whiteblack)
        whiteblackdepth = self.encoderdepth(depth)
        latentImage = self.projectionRGB(torch.flatten(whiteblackencoded,start_dim=1))
        latentDepth = self.projectionDepth(torch.flatten(whiteblackdepth,start_dim=1))
        n_samples = 1
        lstm_input = torch.cat((latentImage,latentDepth,joints),1)
        h_t = torch.zeros(n_samples,self.n_hidden,dtype=torch.float32).to(device)
        c_t = torch.zeros(n_samples,self.n_hidden,dtype=torch.float32).to(device)
        h_t1 = torch.zeros(n_samples,self.n_hidden,dtype=torch.float32).to(device)
        c_t1 = torch.zeros(n_samples,self.n_hidden,dtype=torch.float32).to(device)


        lstm_input = lstm_input[:,None,:]
        for input_t in lstm_input.split(1,dim=0):
            h_t , c_t = self.lstm1(input_t[0],(h_t,c_t))
            h_t1 , c_t1 = self.lstm2(h_t,(h_t1,c_t1))

            output = self.linear1(h_t1)
            output = self.activation(output)
            output = self.linear2(output)
            outputs.append(output)
        outputs = torch.cat(outputs,dim=0)
        return outputs



class reachDataset(Dataset):
    def __init__(self, gray_images, front_disparities,joint_angles):
        self.gray_images = torch.from_numpy(gray_images)
        self.joint_angles = torch.from_numpy(joint_angles)
        self.front_disparities = torch.from_numpy(front_disparities)

    def __len__(self):
        return len(self.gray_images)

    def __getitem__(self, idx):
        gray_image = self.gray_images[idx]
        joint_angle = self.joint_angles[idx]
        front_disparity = self.front_disparities[idx]
        if idx == (len(self.gray_images)-1):
            idx = idx
        else:
            idx = idx+1
        next_joint_angle = self.joint_angles[idx]


        return gray_image,front_disparity,joint_angle,next_joint_angle



def main():
    # ----------------------- SETUP SECTION ---------------------------------------
    # Setup up the simulator
    print('Starting simulation, do not touch the rabbit!')
    # Set headless=False to get an animation. (Slows down learning)
    model = predictivelearning().float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    number_epoch = 10
    batch_size = 4
    env, task = setup_env(headless=False, planning=True)
    demos = task.get_demos(4, live_demos=True)
    while True:
          # -> List[List[Observation]]
        gray_images = []
        front_disparities = []
        joint_angles = []
        for i in range(0,len(demos)):
            rlaction = demos[i]
            for obs in rlaction:
                #gray_image = np.dot(obs.front_rgb[...,:3], [0.2989, 0.5870, 0.1140])
                gray_images.append(obs.front_rgb/255)
                front_disparities.append(obs.front_depth)
                joint_angles.append(obs.joint_positions)
        
        gray_images = np.array(gray_images)
        gray_images = gray_images[:,:,:]
        gray_images = np.transpose(gray_images, (0, 3, 1, 2))
        joint_angles = np.array(joint_angles)
        front_disparities = np.array(front_disparities)
        front_disparities = front_disparities[:,None,:,:]

        dataset = reachDataset(gray_images,front_disparities,joint_angles)
        train_loader = DataLoader(dataset,shuffle=False,batch_size=batch_size)
        for epoch in range(0,number_epoch):
            currentLoss = []
            for data in (train_loader):
                gray_image = data[0].to(device)
                front_depth = data[1].to(device)
                joint_angles = data[2].to(device)
                next_angle = data[3].to(device)
                predicted = model(gray_image.float(),front_depth.float(),joint_angles.float())
                loss = criterion(predicted.double(),next_angle.double())
                loss.backward()
                optimizer.step()
                currentLoss.append(loss)
            print(str(epoch)+".epochnumber"+ "loss:", sum(currentLoss)/len(currentLoss))

        descriptions, obs = task.reset()

        for try_number in range(0,100):
            gray_image = torch.from_numpy(obs.front_rgb[None,:,:,:]/255).to(device).permute(0, 3, 1, 2)
            front_depth = obs.front_depth
            front_depth = torch.from_numpy(np.asarray(front_depth[None,None,:,:])).to(device)
            joint_angle = obs.joint_positions
            joint_angle = torch.from_numpy(joint_angle[None,:]).to(device)

            pred_action = model(gray_image.float(),front_depth.float(),joint_angle.float())

            joint_angle_pred = pred_action.cpu().detach().numpy()
            joint = joint_angle_pred[0]
            action =  np.zeros(8)
            action[0] = joint[0]
            action[1] = joint[1]
            action[2] = joint[2]
            action[3] = joint[3]
            action[4] = joint[4]
            action[5] = joint[5]
            action[6] = joint[6]
            action[7] = 1
            print(try_number)
            obs, reward, terminate = task.step(action)





        



if __name__ == "__main__":
    main()