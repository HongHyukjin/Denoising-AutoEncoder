import copy

def synthesize(noisy, denoised, blind_num):
    img = copy.deepcopy(noisy)
    if blind_num == [0]:
        img[:, 40:80, 40:80, :] = denoised[:, 40:80, 40:80, :]
    elif blind_num == [1]:
        img[:, 80:120, 80:120, :] = denoised[:, 80:120, 80:120, :]
    elif blind_num == [2]:
        img[:, 120:160, 120:160, :] = denoised[:, 120:160, 120:160, :]
    elif blind_num == [3]:
        img[:, 160:200, 160:200, :] = denoised[:, 160:200, 160:200, :]
    elif blind_num == [4]:
        img[:, 200:240, 200:240, :] = denoised[:, 200:240, 200:240, :]
    return img
