from distutils.dir_util import copy_tree

#before_path = r'E:\HyukJin\2021\JongSul\LeeDaeWon\edge_connect\output3'
#after_path = r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\valid7\noisy_image'
before_path = r'E:\HyukJin\2021\JongSul\LeeDaeWon\edge_connect\datasets\paris_train_original(5000)_256'
after_path = r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\valid7\reference'

copy_tree(before_path, after_path)