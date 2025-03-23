import retro
from flexibuff import FlexibleBuffer
from flexibuddiesrl import TD3, DQN, PG
import numpy as np
import pyglet
from inputs import get_gamepad  # For handling input devices (controllers, keyboard, etc.)
from inputs import devices
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os

env = retro.make(game="MortalKombat3-Genesis")  

ROW_CENTER = 67
COL_CENTER = 120

cur_row_center = ROW_CENTER  # Initialize the current row center for the environment (if needed)
cur_col_center = COL_CENTER  # Initialize the current column center for the environment (if needed)
#env.viewer = # Ensure the viewer is initialized for rendering (if needed)
obs = env.reset()
env.render(mode='human')
window = env.viewer.window
window.set_size(320*3,224*3)  # Set the window size to match the environment's resolution (if needed)

global_action = np.zeros(12)  # Global action array to store the current action to be passed to the environment
actions = ['b','a*','nothing','nothing','up','down','left','right','c*','y','x*','z*']  # List to store the actions based on button statuses
#           0   1     2          3        4    5      6      7      8    9   10   11
btn_mapping = {
    "BTN_WEST": 0,#1,  # Typically the 'X' button on many controllers (e.g., PS, Xbox)
    "BTN_SOUTH": 1,#0,  # Typically the 'Y' button on many controllers (e.g., PS, Xbox)
    "BTN_EAST": 10,#8,  # Typically the 'Y' button on many controllers (e.g., PS, Xbox)

    "BTN_TL": 11,#10,
    "BTN_TR": 8,#9,  # Right trigger on many controllers (e.g., PS, Xbox)
    "BTN_NORTH": 9,#11,
    # Add other buttons as needed (e.g., x, y, z)
}
nn_button_key = [0,1,8,9,10,11,2]
nn_direction_key = np.array([
        [1,0,0,0], # up 
        [1,0,0,1], # up+right
        [0,0,0,1], # right
        [0,1,0,1], # down+right
        [0,1,0,0], # down
        [0,1,1,0], # down+left
        [0,0,1,0], # left 
        [1,0,1,0], # up+left
        [0,0,0,0], # no-op
    ])
direction_to_nn_key = [
    "1000", # up 
    "1001", # up+right
    "0001", # right
    "0101", # down+right
    "01010", # down
    "0110", # down+left
    "0010", # left 
    "1010", # up+left
    "0000", # no-op
]
absx = 0
absy = 0

# if window:
#     @window.event
#     def on_key_press(key, mod):
        
#         """
#         Callback to handle keypress events.
#         :param key: Key pressed
#         :param mod: Modifier keys (shift, ctrl, etc.)
#         :return: None
#         """
#         print(f"Key pressed: {key} with modifier {mod}")
#         pass

#     @window.event
#     def on_key_release(key, mod):
#         """
#         Callback to handle key release events.
#         :param key: Key released
#         :param mod: Modifier keys (shift, ctrl, etc.)
#         :return: None
#         """
#         print(f"Key released: {key} with modifier {mod}")
#         pass

def get_direction(x, y):
    """
    Converts x and y joystick positions into 8-directional movement.
    :param x: X-axis value (-32000 to 32000)
    :param y: Y-axis value (-32000 to 32000)
    :return: [up,down,left,right] 1 if true, 0 if not
    """
    # Define thresholds to avoid sensitivity issues
    threshold = 10000  # Deadzone threshold to ignore small movements

    # Normalize x and y to -1, 0, or 1 based on the threshold

    right = 1 if x > threshold else 0
    left = 1 if x < -threshold else 0
    up = 1 if y > threshold else 0
    down = 1 if y < -threshold else 0

    return right, left, up, down  # Return the normalized values for x and y    
   
def handle_controller_input():
    """
    Polls the controller for input and processes button presses and axis motion.
    """
    while True:
        try:
            events = get_gamepad()
            for event in events:
                if event.ev_type == "Key":
                    #print(f"Button released: {event.code}")
                    if event.state:  # Button pressed
                        #print(event.code)
                        global_action[btn_mapping[event.code]] = 1.0  # Mark the button as pressed
                    else:  # Button released
                        #print(f"Button released: {event.code}")
                        global_action[btn_mapping[event.code]] = 0  # Mark the button as pressed

                elif event.ev_type == "Absolute":  # Axis motion
                    #print(f"Axis {event.code} moved to {event.state}")
                    if event.code=="ABS_X":
                        absx = event.state
                    elif event.code=="ABS_Y":
                        absy = event.state
                    
                    # Get the 8-direction based on the current x and y values
                    right,left,up,down = get_direction(absx, absy)  # Get the direction based on x and y values
                    global_action[4] = up   # up
                    global_action[5] = down # down
                    global_action[6] = left # left                    
                    global_action[7] = right # right
        
           # print("Current button statuses: ", button_statuses)  # Print the current status of all buttons
        except Exception as e:
            """
            Handle exceptions that may occur when polling the gamepad.
            This could happen if no gamepad is connected or if an unknown event occurs.
            """
            print(f"Error handling controller input: {e}")
        pass

def nn_to_action(nn_action):
    action = np.zeros(12)
    action[4:8] = nn_direction_key[nn_action[0]]  # up,down,left,right based on the first 4 bits of the nn_action
    action[nn_button_key[nn_action[1]]] = 1.0  # Set the button based on the second part of the nn_action (button index)
    return action

def action_to_nn(action):
    nn_action = np.zeros(2)
    for i,button in enumerate([0,1,8,9,10,11,2]):
        if action[button] == 1.0:
            nn_action[1] = i
            break
    # Now find the direction
    dir_string = np.clip(action[4:8], 0, 1)  # Ensure the direction is binary (0 or 1)
    dir_string = ''.join(str(int(x)) for x in dir_string)  # Convert to string format (e.g., '1000', '0101')
    # Find the index in the direction_to_nn_key that matches this string
    for idx, key in enumerate(direction_to_nn_key):
        if key == dir_string:
            nn_action[0] = idx
            break
    return nn_action.astype(np.int32)  # Return the final nn_action as an integer array


def pil_down(obs):
    #downscale obs image by 2 with Pillow
    x = PIL.Image.fromarray(obs)  # Convert the numpy array to a PIL Image
    x = x.resize((int(obs.shape[1] / 2), int(obs.shape[0] / 2)),Image.LANCZOS)  # Downsample the image by 2x using Pillow (LANCZOS is a good downsampling filter)
    return x

def cv2_down(obs, n=2):
    #downsample obs image by 2 using cv2
    x = cv2.resize(obs, (int(obs.shape[1] / n), int(obs.shape[0] / n)), interpolation=cv2.INTER_LINEAR) # Downsample the image by 2x using cv2
    #print(f"cv2_downsampled shape: {x.shape}")  # Debug: Check the downsampled image shape
    return x

def preprocess_image(obs, n=2, step=0, ims=None):
    x = cv2_down(obs, n)
    color = np.array([232,168,136],dtype=np.uint8)#[230,200,170]#[232,204,168]
    x2 = cv2.inRange(x,lowerb=color-17,upperb=color+17)
    
    x = np.sum(x//3, axis=-1) # Convert to grayscale if needed (optional, depending on your use case)

    if step == 0:
        ims[1] = x # if step is zero, fill the im buffer with this observation
        ims[2] = x 
    ims[step%3] = x # store the image for the correct step in the buffer
    
    x = np.stack([ims[step%3], ims[(step-1)%3], ims[(step-2)%3]], axis=0) 

    return x

def main(agent:DQN=None, 
         rl_memory:FlexibleBuffer=None, 
         human_memory:FlexibleBuffer=None, 
         use_rl=False, 
         use_sl=False,
         name='default_player',
         recording =False,
         downsize=2,
         ):
    ims = np.zeros((3,224//downsize,320//downsize), dtype=np.uint8)  # Placeholder for the image to be processed

    try:
        gamepadstuff = get_gamepad()
        input_thread = threading.Thread(target=handle_controller_input)
        input_thread.daemon = True  # Daemonize thread to exit when the main program exits
        input_thread.start()  # Start the thread to handle controller input
    except Exception as e:
        print("did not work",e)
    
    raw_obs = env.reset()
    obs = preprocess_image(raw_obs, n=downsize, step=0, ims=ims)  # Preprocess the initial observation (downsample and convert to grayscale if needed)
    step = 0
    #act_test = env.action_space.sample()\
    raw_action = np.zeros(2)  # nn action format
    ep_rews = [0]  # Initialize episode reward (if needed for logging or tracking)
    while True:
        if recording:
            start_time = time.time()  # Start time for recording the episode (if needed)
        if not recording:
            T_obs = torch.tensor(obs, dtype=torch.float32).to(encoder.device).unsqueeze(0)  # Convert the observation to a tensor
            nn_action, _, _1, _2, _3 = agent.train_actions(T_obs,step=True)  # Get the action from the agent, if provided, otherwise use zeros
            raw_action = nn_action
            action = nn_to_action(nn_action)
        else:
            action = np.copy(global_action)
            raw_action = action_to_nn(action)  # Convert the global action to the neural network's format

        action[2] = 0
        action[3] = 0  # Ensure 'nothing' is set to 0 for the environment (if needed)

        surrogate_action = np.zeros(12)
        #if action[0]>0.1:
        surrogate_action[0] = action[0]
        surrogate_action[4:8] = action[4:8]
        _1,       _2,   _3,  _4,  = env.step(surrogate_action)  # Perform the action in the environment, this will return the next observation, reward, done, and info (if any)
        raw_obs_, rew, done, info = env.step(action)
        rew = rew+_2


        ep_rews[-1] += rew  # Accumulate the reward for the current episode (if needed for logging or tracking)
        obs_ = preprocess_image(raw_obs_, n=downsize, step=step, ims=ims)  # Preprocess the next observation (downsample and convert to grayscale if needed)
        if step == 12999:
            done = True  # Force end the episode after 5000 steps to not overfill memory buffer
        if rl_memory is not None and not recording:
            # Store the transition in the memory buffer, if provided
            rl_memory.save_transition(terminated=done,registered_vals = { 
                    "global_rewards": rew,
                    "obs": obs,
                    "obs_": obs_,
                    "discrete_actions": raw_action,  # Store the neural network's action (discrete) for the agent
                })
        
        if name != 'default_player' and human_memory is not None and recording:
            human_memory.save_transition(terminated=done,registered_vals = { 
                    "global_rewards": rew,
                    "obs": obs,
                    "obs_": obs_,
                    "discrete_actions": raw_action,  # Store the neural network's action (discrete) for the agent
                })

        if step%4 == 0 and not recording:  # Perform learning every 8 steps (or adjust as needed)
            if use_rl and agent is not None and rl_memory.steps_recorded >= 64:  # Ensure we have enough samples in the buffer to learn from
                # Perform a learning step for the RL agent, if applicable
                rl_batch = rl_memory.sample_transitions(batch_size=64, as_torch=True, device='cuda')  # Sample a batch from the RL memory buffer, if available
                rl_batch.obs = rl_batch.obs.float()
                rl_batch.obs_ = rl_batch.obs_.float()
                rl_batch.discrete_actions = rl_batch.discrete_actions.to(torch.int64)  # Ensure the discrete actions are in the correct format for the agent
                #print(rl_batch.obs.shape)  # Debug: Check the contents of the sampled batch before learning
                rl_loss, _ = agent.reinforcement_learn(rl_batch)
                
            if use_sl and agent is not None and human_memory.steps_recorded >= 64:  # Ensure we have enough samples in the human memory buffer to learn from
            #if rl_memory.steps_recorded>=64:   
                # Perform a learning step for the SL agent, if applicable
                sl_batch = human_memory.sample_transitions(batch_size=64, as_torch=True, device='cuda')
                sl_batch.obs = sl_batch.obs.float()
                sl_batch.obs_ = sl_batch.obs_.float()
                sl_batch.discrete_actions = sl_batch.discrete_actions.to(torch.int64)  # Ensure the discrete actions are in the correct format for the agent
                #print(rl_batch.obs.shape)  # Debug: Check the contents of the sampled batch before learning
                sl_loss, _ = agent.imitation_learn(sl_batch.obs[0],sl_batch.discrete_actions[0])
        step+=1

        if rew>0:
            print(f"{step}: rew={rew:.4f} ")
            print(agent.eps)  # Print the current epsilon value for exploration (if applicable, e.g., in DQN or TD3)
        env.render()

        if recording and time.time() - start_time < (1/30):  # Limit recording to 60 seconds (if needed for performance)
            time.sleep((1/30) - (time.time() - start_time))
        if done:
            raw_obs = env.reset()  # Reset the environment if done
            obs = preprocess_image(raw_obs, n=downsize, step=0, ims=ims)  # Preprocess the new observation after reset
            step=0
            ep_rews.append(0)  # Reset the episode reward for the new episode (if needed for logging or tracking)
            print(ep_rews)
            #plt.plot(ep_rews)  # Plot the episode rewards over time (optional, can be removed)
            #plt.show()
            if recording:
                print(f"Finished recording an episode, total rewards: {ep_rews[-1]}")
                FlexibleBuffer.save(human_memory)  # Save the human memory buffer after recording an episode (if applicable)
            agent.save(f"./{name}_model")  # Save the model after each episode (if applicable)
            if human_memory.steps_recorded > 14500:
                exit()

        obs = obs_  # Update the current observation to the next one for the next iteration
    env.close()


class cnn_encoder(nn.Module):
    def __init__(self, obs_dim=10, device='cpu'):
        super(cnn_encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=6, stride=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.lin1 = nn.Linear(960,64)
        self.to(device)
        self.device=device

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.lin1(x.reshape(x.shape[0], -1)))  # Apply the linear layer after flattening the conv output, if needed (960 is from conv3 output)
        #x = #x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # Flatten the output for the fully connected layer
        # print(f"Shape after conv layers: {x.shape}")  # Debug: Check the shape after convolutional layers
        return x  # Flatten the output for the fully connected layer

if __name__ == "__main__":
    N = 2  # Downsampling factor (change as needed, e.g., 2, 4, etc.)
    #argparse for player name, RL true or false, and SL true or false
    parser = argparse.ArgumentParser(description='Run a Retro game with RL agent')
    parser.add_argument('--rl', action='store_true', help='Use RL agent (DQN, TD3, etc.)')
    parser.add_argument('--sl', action='store_true', help='Use SL agent (e.g., PG)')
    parser.add_argument('--name', type=str, default='default_player', help='Player name for the agent')
    parser.add_argument('--recording', action='store_true', help='Enable recording mode (saves transitions to memory without learning)')
    args = parser.parse_args()

    # agent = TD3(obs_dim=
    #agent = TD3(obs_dim=244*320*3,
    #            discrete_action_dims=(12,))
    encoder = cnn_encoder(device='cuda')  # Create an instance of the encoder, if needed for TD3 or DQN
    DQN_agent = DQN(obs_dim=0,
                    discrete_action_dims=[9,7],#np.ones(12,dtype=np.int64)+1,  # 12 discrete actions for the buttons
                    continuous_action_dims=0,
                    encoder=encoder,
                    hidden_dims=[64],#[4928],
                    gamma=0.995,
                    dueling=True,
                    device='cuda',
                    eps_decay_half_life=30000,
                    entropy=0,
                    munchausen=0,
                    lr=3e-4)  # Pass the encoder if needed for the agent
    #DQN_agent.load(f"./{args.name}_model")  # Load the model if it exists (optional, for resuming training or evaluation)
    rl_memory = FlexibleBuffer(num_steps=20000,
                               n_agents=1,
                               discrete_action_cardinalities=[9,7],
                               track_action_mask=False,
                               path="rlmemory",  # Path to save the memory buffer (optional, for persistence)
                               name="test1",
                               global_registered_vars={ 
                                   "global_rewards": (None, np.float32),
                                   },
                               individual_registered_vars={
                                   "obs": ([3,224//N,320//N], np.uint8),
                                   "obs_": ([3,224//N,320//N], np.uint8), 
                                   "discrete_actions": ([2], np.uint8),
                               },
                            ) # Size of the memory buffer for storing transitions)
    human_memory = FlexibleBuffer(num_steps=10000,
                               n_agents=1,
                               discrete_action_cardinalities=[9,7],
                               track_action_mask=False,
                               path="./"+args.name+'/',  # Path to save the memory buffer (optional, for persistence)
                               name="memory",
                               global_registered_vars={ 
                                   "global_rewards": (None, np.float32),
                                   },
                               individual_registered_vars={
                                   "obs": ([3,224//N,320//N], np.uint8),
                                   "obs_": ([3,224//N,320//N], np.uint8), 
                                   "discrete_actions": ([2], np.uint8),},
                            )
    if os.path.exists("./"+args.name+ "/"):
        human_memory = FlexibleBuffer.load("./"+args.name+'/', "memory")  # Load the human memory buffer if it exists (optional, for resuming training or evaluation)
        print(human_memory)
        #exit()
        #FlexibleBuffer.save(human_memory)
    
    #https://github.com/openai/retro/issues/33
    #https://ocw.mit.edu/courses/14-126-game-theory-spring-2016/25c82e8213fd23daa72c9382e921a2b0_MIT14_126S16_cooperative.pdf
    main(DQN_agent, rl_memory, human_memory,args.rl,args.sl, args.name, args.recording, downsize=N)  # Start the main loop with the agent, if using an agent like TD3 or DQN
