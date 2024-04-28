import time
from progressbar import *
from hnsw import HNSW
import numpy as np

#维度20、数据量1W
dim = 20
num_elements = 10000

#随机生成1W个数据、m0：第0层最大连接数、ef：候选集数
data = np.array(np.float32(np.random.random((num_elements, dim))))
hnsw = HNSW('l2', m0=16, ef=64)
widgets = ['HNSW build progress: ',Percentage(), ' ', Bar('#'),' ', Timer(), ' ', ETA()]

# show progressbar
pbar = ProgressBar(widgets=widgets, maxval=num_elements).start()# 进度条
for i in range(len(data)):
    hnsw.add(data[i])
    pbar.update(i + 1)
pbar.finish()


search_data = np.float32(np.random.random((1, dim)))# 目标节点
start_time = time.time()
min_distance = 1000000
min_idx = -1
for i in range(num_elements):
    check_data = data[i,:]
    tmp_distance = np.sqrt(np.sum((check_data- search_data) ** 2,axis=1))[0]
    if tmp_distance < min_distance:
        min_distance = tmp_distance
        min_idx = i
end_time = time.time()
print("brute force => Searchtime: %f, result idx: %d, min_distance : %f" % ((end_time - start_time),min_idx,min_distance))

add_point_time = time.time()
[idx,dis] = hnsw.search(search_data, 1)[0]
search_time = time.time()
print("use hnsw    => Searchtime: %f, result idx: %d, min_distance : %f" % ((search_time - add_point_time),idx,dis))

