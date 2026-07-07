import numpy as np

data = np.load("speed_tensor.npz")
speed = data["speed"]
print("张量形状:", speed.shape)
mask = data["obs_mask"]
print("观测掩码形状:", mask.shape)
segs = data["segs"]
print("路段数量:", len(segs))
print("路段编号:", segs)

# # 随便抽一个路段、某天、某个时段查看
# seg_id = 0
# day_id = 5
# slot_id = 12
# print("速度值：", speed[seg_id, day_id, slot_id])
# print("是否有观测：", mask[seg_id, day_id, slot_id])