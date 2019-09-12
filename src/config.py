import os


# ----------------------------------------------------------------------------------------------------------------------
# Dataset Settings
#
train_path = os.path.abspath(os.path.join(__file__, "../..", "Dataset/Jester2_train.csv"))
valid_path = os.path.abspath(os.path.join(__file__, "../..", "Dataset/Jester2_valid.csv"))
test_path = os.path.abspath(os.path.join(__file__, "../..", "Dataset/Jester2_test.csv"))


# ----------------------------------------------------------------------------------------------------------------------
# Model Settings
#
dim_factors = 20


# ----------------------------------------------------------------------------------------------------------------------
# Quantization Settings
#
#  0 ---- 1 ---- 2 ---- 3 ---- 4 ---- 5 ---- =>
#  |  q0  |  q1  |  q2  |  q3  |  q4  |  q5  =>
quantize_unit = 1
max_quantize = 10


# ----------------------------------------------------------------------------------------------------------------------
# Training Settings
#
learning_rate = 1e-6
emv_decay = 0.998
batch_size = 128
max_epoch = 20


# ----------------------------------------------------------------------------------------------------------------------
# Testing Settings
#
top_ks = [5, 10, 15, 20]
