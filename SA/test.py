
pickle_data_path = "region_grid_map.pkl"

import pickle
f = open(pickle_data_path,'rb')   #pickle_data_path为.pickle文件的路径；
info = pickle.load(f)
print(info)
f.close()  #别忘记close pickle文件
