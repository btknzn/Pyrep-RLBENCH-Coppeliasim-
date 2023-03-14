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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset





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



class encoderBlack(nn.Module):
    def __init__(self):
        super(encoderBlack, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5,stride=2),
            nn.ReLU(),
            #nn.BatchNorm2d(8),
            nn.Conv2d(4,16,kernel_size=5,stride=2),
            nn.ReLU(),
            #nn.BatchNorm2d(16),
            nn.Conv2d(16,32,kernel_size=5,stride=2),
            nn.ReLU(),
            #nn.BatchNorm2d(64),
            )
    def forward(self,x):
        x = self.encoder(x)
        return x
    
class encoderRGB(nn.Module):
    def __init__(self):
        super(encoderRGB, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=5,stride=2),
            nn.ReLU(),
            #nn.BatchNorm2d(8),
            nn.Conv2d(4,16,kernel_size=5,stride=2),
            nn.ReLU(),
            #nn.BatchNorm2d(16),
            nn.Conv2d(16,64,kernel_size=5,stride=2),
            nn.ReLU(),
            #nn.BatchNorm2d(64),
            )
    def forward(self,x):
        x = self.encoder(x)
        return x
    

class predictivelearning(nn.Module):
    def __init__(self,n_hidden=1024,out_dim=7):
        super(predictivelearning, self).__init__()
        self.n_hidden = n_hidden
        # projectionRGB + projectionDepth+7
        self.lstm1 = nn.LSTMCell(295,self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden,self.n_hidden)
        self.lstm3 = nn.LSTMCell(self.n_hidden,self.n_hidden)
        self.lstm4 = nn.LSTMCell(self.n_hidden,self.n_hidden)
        self.lstm5 = nn.LSTMCell(self.n_hidden,self.n_hidden)
        

        self.encoderdepth1 = encoderBlack()
        self.encoderRGB1 = encoderRGB()
        self.encoderdepth2 = encoderBlack()
        self.encoderRGB2 = encoderRGB()
        self.encoderdepth3 = encoderBlack()
        self.encoderRGB3 = encoderRGB()

        self.linear1 = nn.Linear(self.n_hidden,32)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(32,out_dim)

    def forward(self,rgb1,rgb2,rgb3,black1,black2,black3,joints):
        outputs = []
        RGBencoded1 = (torch.flatten(self.encoderRGB1(rgb1),start_dim=1))
        RGBencoded2 =(torch.flatten(self.encoderRGB2(rgb2),start_dim=1))
        RGBencoded3 = (torch.flatten(self.encoderRGB3(rgb3),start_dim=1))
        Blackencoded1 = (torch.flatten(self.encoderdepth1(black1),start_dim=1))
        Blackencoded2 = (torch.flatten(self.encoderdepth1(black2),start_dim=1))
        Blackencoded3 = (torch.flatten(self.encoderdepth1(black3),start_dim=1))
        #whiteblackdepth = self.encoderdepth(depth)
        #latentImage = self.projectionRGB(torch.flatten(whiteblackencoded,start_dim=1))
        #latentDepth = self.projectionDepth(torch.flatten(whiteblackdepth,start_dim=1))
        n_samples = 1
        lstm_input = torch.cat((RGBencoded1,RGBencoded2,RGBencoded3,Blackencoded1,Blackencoded2,Blackencoded3,joints),1)
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
    def __init__(self, front_rgbs,left_shoulder_rgbs,right_shoulder_rgbs,front_disparities,left_shoulder_disparities,right_shoulder_disparties,joint_angles):
        self.front_rgbs = torch.from_numpy(front_rgbs)
        self.left_shoulder_rgbs = torch.from_numpy(left_shoulder_rgbs)
        self.right_shoulder_rgbs = torch.from_numpy(right_shoulder_rgbs)
        self.joint_angles = torch.from_numpy(joint_angles)
        self.front_disparities = torch.from_numpy(front_disparities)
        self.left_shoulder_disparities =  torch.from_numpy(left_shoulder_disparities)
        self.right_shoulder_disparties = torch.from_numpy(right_shoulder_disparties)

    def __len__(self):
        return len(self.front_rgbs)

    def __getitem__(self, idx):
        front_rgb = self.front_rgbs[idx]
        left_shoulder_rgb = self.left_shoulder_rgbs[idx]
        right_shoulder_rgb = self.right_shoulder_rgbs[idx]
        front_disparity = self.front_disparities[idx]
        left_disparity = self.left_shoulder_disparities[idx]
        right_disparity = self.right_shoulder_disparties[idx]

        joint_angle = self.joint_angles[idx]
        if idx == (len(self.front_rgbs)-1):
            idx = idx
        else:
            idx = idx+1
        next_joint_angle = self.joint_angles[idx]


        return front_rgb,left_shoulder_rgb,right_shoulder_rgb ,front_disparity,left_disparity,right_disparity,joint_angle,next_joint_angle



def main():
    # ----------------------- SETUP SECTION ---------------------------------------
    # Setup up the simulator
    print('Starting simulation, do not touch the rabbit!')
    # Set headless=False to get an animation. (Slows down learning)
    model = predictivelearning().float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    number_epoch = 100
    batch_size = 64
    env, task = setup_env(headless=False, planning=True)
    while True:
        demos = task.get_demos(10, live_demos=True)
        front_rgbs = []
        left_shoulder_rgbs = []
        right_shoulder_rgbs = []
        front_disparities = []
        left_shoulder_disparities = []
        right_shoulder_disparties = []
        joint_angles = []
        for i in range(0,len(demos)):
            rlaction = demos[i]
            for obs in rlaction:
                front_rgb = np.resize((obs.front_rgb/255), (32,32,3))
                front_rgbs.append(front_rgb)
                left_should_rgb = np.resize((obs.left_shoulder_rgb/255), (32, 32,3))
                left_shoulder_rgbs.append(left_should_rgb)
                right_shoulder_rgb = np.resize((obs.right_shoulder_rgb/255),(32, 32,3))
                right_shoulder_rgbs.append(right_shoulder_rgb)
                front_disparity = np.resize((obs.front_depth),(32, 32))
                front_disparities.append(front_disparity)
                left_shoulder_disparity = np.resize((obs.left_shoulder_depth),(32, 32))
                left_shoulder_disparities.append(left_shoulder_disparity)
                right_shoulder_disparity = np.resize((obs.right_shoulder_depth),(32, 32))
                right_shoulder_disparties.append(right_shoulder_disparity)
                joint_angles.append(obs.joint_positions)
        
        front_rgbs = np.array(front_rgbs)
        front_rgbs = front_rgbs[:,:,:]
        front_rgbs = np.transpose(front_rgbs, (0, 3, 1, 2))
        left_shoulder_rgbs = np.array(left_shoulder_rgbs)
        left_shoulder_rgbs = left_shoulder_rgbs[:,:,:]
        left_shoulder_rgbs = np.transpose(left_shoulder_rgbs, (0, 3, 1, 2))
        right_shoulder_rgbs = np.array(right_shoulder_rgbs)
        right_shoulder_rgbs = right_shoulder_rgbs[:,:,:]
        right_shoulder_rgbs = np.transpose(right_shoulder_rgbs, (0, 3, 1, 2))
        
        front_disparities = np.array(front_disparities)
        front_disparities = front_disparities[:,None,:,:]
        left_shoulder_disparities = np.array(left_shoulder_disparities)
        left_shoulder_disparities = left_shoulder_disparities[:,None,:,:]
        right_shoulder_disparties = np.array(right_shoulder_disparties)
        right_shoulder_disparties = right_shoulder_disparties[:,None,:,:]

        joint_angles = np.array(joint_angles)
        dataset = reachDataset(front_rgbs,
        left_shoulder_rgbs,right_shoulder_rgbs,front_disparities,left_shoulder_disparities,right_shoulder_disparties,joint_angles)
        train_loader = DataLoader(dataset,shuffle=False,batch_size=batch_size)
        for epoch in range(0,number_epoch):
            currentLoss = []
            for data in (train_loader):
                next_angle = data[7].to(device)
                predicted = model(data[0].to(device).float(),data[1].to(device).float(),data[2].to(device).float(),data[3].to(device).float(),data[4].to(device).float(),data[5].to(device).float(),data[6].to(device).float())
                loss = criterion(predicted.double(),next_angle.double())
                loss.backward()
                optimizer.step()
                currentLoss.append(loss)
            print(str(epoch)+".epochnumber"+ "loss:", sum(currentLoss)/len(currentLoss))

        descriptions, obs = task.reset()
        torch.save(model.state_dict(), 'model.pth')
        for try_number in range(0,100):
            front_image = torch.from_numpy(np.resize(obs.front_rgb[None,:,:,:]/255,(32,32,3))[None,:,:,:]).to(device).permute(0, 3, 1, 2)
            left_shoulder_image = torch.from_numpy(np.resize(obs.left_shoulder_rgb[None,:,:,:]/255,(32,32,3))[None,:,:,:]).to(device).permute(0, 3, 1, 2)
            right_shoulder_image = torch.from_numpy(np.resize(obs.right_shoulder_rgb[None,:,:,:]/255,(32,32,3))[None,:,:,:]).to(device).permute(0, 3, 1, 2)
            front_depth = np.resize(obs.front_depth,(32,32))
            front_depth = torch.from_numpy(np.asarray(front_depth[None,None,:,:])).to(device)
            left_shoulder_depth = np.resize(obs.left_shoulder_depth,(32,32))
            left_shoulder_depth = torch.from_numpy(np.asarray(left_shoulder_depth[None,None,:,:])).to(device)
            right_shoulder_depth = np.resize(obs.right_shoulder_depth,(32,32))
            right_shoulder_depth = torch.from_numpy(np.asarray(right_shoulder_depth[None,None,:,:])).to(device)
            joint_angle = obs.joint_positions
            joint_angle = torch.from_numpy(joint_angle[None,:]).to(device)
            
            pred_action = model(front_image.float(),left_shoulder_image.float(),right_shoulder_image.float(),
                                front_depth.float(),left_shoulder_depth.float(),right_shoulder_depth.float(),joint_angle.float())

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