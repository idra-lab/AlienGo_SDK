import torch

def swap_legs(array):
    """
    Swap the front and rear legs of the array based on predefined indices.
    
    The swap logic is fixed:
    - Swap front legs (indices 3:6) with (0:3)
    - Swap rear legs (indices 9:12) with (6:9)
    """
    array_copy = array.copy()  # Make a copy to avoid modifying the original array
    order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    return array_copy[order]

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