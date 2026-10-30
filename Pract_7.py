import numpy as np
import tensorflow as tf

def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000,
        (2 * (np.arange(d_model)[np.newaxis, :] // 2))
        / np.float32(d_model)
    )

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


position = 50
d_model = 512

pos_encoding = positional_encoding(position, d_model)

print("Positional Encodings Shape:", pos_encoding.shape)
print("Positional Encodings Example:\n", pos_encoding)


import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


input_data = torch.randn(10, 64)
model = SimpleNN(64, 10)
output = model(input_data)

print(output)
