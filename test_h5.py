import h5py
import numpy as np
with h5py.File('./data_sets/test_catvnoncat.h5',"r") as f:
    for key in f.keys():
    	 #print(f[key], key, f[key].name, f[key].value) # 因为这里有group对象它是没有value属性的,故会异常。另外字符串读出来是字节流，需要解码成字符串。
        print(f[key], key, f[key].name) # f[key] means a dataset or a group object. f[key].value visits dataset' value,except group object.

    dogs_group = f["list_classes"] # 从上面的结果可以发现根目录/下有个dogs的group,所以我们来研究一下它
    
    print(np.array(dogs_group[:]))