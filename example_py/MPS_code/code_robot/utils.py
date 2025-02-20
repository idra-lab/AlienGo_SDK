# utils.py
import numpy as np
import torch
from collections import OrderedDict

# Quaternion rotation helper
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def scale_axis(index, value):
    """
    Scales the input value based on the specified axis index.

    Parameters:
    index (int): The index of the axis to scale. 
                 0 - Axis 1 (Left Stick Y)
                 1 - Axis 0 (Left Stick X)
                 2 - Axis 3 (Trigger)
    value (float): The input value to be scaled.

    Returns:
    float: The scaled value based on the axis index.
           - For index 0: The value is flipped in sign and scaled to the range [-0.5, 0.5].
           - For index 1: The value is flipped in sign and scaled to the range [0, 1.5] for positive values and [0, -0.9] for negative values.
           - For index 2: The value is scaled symmetrically to the range [-0.78, 0.78].
           - For other indices: The value is returned without scaling.
    """
    if index == 0:  
        return -value * 0.5 
    elif index == 1: 
        value *= -1 
        if value > 0:
            return value * 1.5  
        else:
            return value * 0.9  
    elif index == 2: 
        return value * 0.78 
    else:
        return value  # Default case, no scaling for other axes

def swap_legs(array):
    """
    Swap the front and rear legs of the array based on predefined indices.
    
    The swap logic is fixed:
    - Swap front legs (indices 3:6) with (0:3)
    - Swap rear legs (indices 9:12) with (6:9)
    """
    array_copy = array.copy()  # Make a copy to avoid modifying the original array
    
    # Swap front legs (3:6) with (0:3)
    array_copy[0:3] = array[3:6]
    array_copy[3:6] = array[0:3]
    
    # Swap rear legs (9:12) with (6:9)
    array_copy[6:9] = array[9:12]
    array_copy[9:12] = array[6:9]
    
    return array_copy


def clip_torques_in_groups(torques):
    """
    Clip the elements of the `torques` array in groups of 3 with different ranges for each element.
    - The first and second elements in the group are clipped to [-35.0, 35.0]Nm
    - The third element in the group is clipped to [-45.0, 45.0]Nm
    """
    torques_copy = torques.copy()  # Make a copy to avoid modifying the original array

    # Iterate over the array in groups of 3
    for i in range(0, len(torques), 3):  # Step by 3 to handle each group
        torques_copy[i] = np.clip(torques_copy[i], -35.0, 35.0)         # First element in group
        torques_copy[i + 1] = np.clip(torques_copy[i + 1], -35.0, 35.0)   # Second element in group
        torques_copy[i + 2] = np.clip(torques_copy[i + 2], -45.0, 45.0)   # Third element in group

    return torques_copy

def labels_state_dict(old_state_dict, old_keys, new_keys):
    new_state_dict = OrderedDict()
    new_dict = 0
    for k, v in old_state_dict.items():
        if k in old_keys:
            name = new_keys[new_dict] # remove `module.`
            new_state_dict[name] = v
            new_dict += 1
    return new_state_dict

def order_state(state):#pos, vel):
    order_pos = [10, 7, 16, 13, 11, 8, 17, 14, 12, 9, 18, 15]
    order_vel = [ 9, 6, 15, 12, 10, 7, 16, 13, 11, 8, 17, 14]
    return np.concatenate((pos[0:7], vel[0:6], pos[order_pos], vel[order_vel]))

def order_backup(pos):
    order_pos = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
    return pos[order_pos]#.detach().cpu().numpy()