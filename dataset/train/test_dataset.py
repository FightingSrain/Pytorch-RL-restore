
import h5py as h5
train_file = h5.File("train.h5", "r")
data = train_file["data"].value
label = train_file["label"].value
# print(data.shape)
# print(label.shape)
# (2752, 63, 63, 3) data
# (2752, 63, 63, 3) label
data_index = 3
img = data[data_index: data_index+1, ...]
print(img.shape)
print(len(data))