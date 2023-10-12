import torch
from PIL import Image, ImageChops
import numpy as np

def cal_max_min(log_history):
    max_value, min_value = max(log_history), min(log_history)
    return max_value - min_value

def get_split_list(start, end, num, dtype=np.int32):
    return np.linspace(start, end, num, endpoint=True, dtype=dtype).tolist()



def cosine_sim_layer(gradient):

    def cal_cosine(a, b):
        scalar_product = (a * b).sum() 
        rec_norm = a.pow(2).sum()
        data_norm = b.pow(2).sum()

        objective = 1 - scalar_product / ((rec_norm.sqrt() * data_norm.sqrt()) + 1e-6)
        
        return objective

    


    cos_0_1 = cal_cosine(gradient[:, 0, :, :],gradient[:, 1, :, :] )
    cos_0_2 = cal_cosine(gradient[:, 0, :, :],gradient[:, 2, :, :] )
    cos_1_2 = cal_cosine(gradient[:, 1, :, :],gradient[:, 2, :, :] )

    # print(cos_0_1, cos_0_2, cos_1_2)

    
    return (cos_0_1 + cos_0_2 + cos_1_2)/3
def save_img(dir, img_tensor, iteration=0, trial=1, is_batch=True, dm=0, ds=1):
    '''save torch tensor to img
    
    : dir save_img dir
    :img_tensor img tensor 
    :iteration iteration
    : is_batch  is img_tensor includes batch dimension
    : dm dataset mean in each channel [1,3,1,1]
    : ds dataset stard variation in each channel [1,3,1,1]
    
    '''
    import torchvision
    import os

    if not os.path.exists(dir):
        #make dirs
        os.makedirs(dir, exist_ok=True)
        # os.makedirs(dir)

    trial_path = dir + '/' + f'{trial}' 
    if not os.path.exists(trial_path):
        os.mkdir(trial_path)

    img_tensor = torch.clamp(img_tensor * ds + dm, 0, 1)
    if is_batch:
       for  i, img in enumerate(img_tensor):

            img = torchvision.transforms.ToPILImage()(img)
            path = trial_path + '/' + f'{iteration}_{i}.png' 
            img.save(path)
    else:
        # img.mul_(ds).add_(dm).clamp_(0, 1)
        # img = torchvision.transforms.ToPILImage()(img_tensor)
        # img.save(path)
        raise Exception('Except batch dimension in img tensor')

def save_img_d(path, img_tensor, is_batch=True, dm=0, ds=1):
    '''save torch tensor to img
    
    : dir save_img dir
    :img_tensor img tensor 
    :iteration iteration
    : is_batch  is img_tensor includes batch dimension
    : dm dataset mean in each channel [1,3,1,1]
    : ds dataset stard variation in each channel [1,3,1,1]
    
    '''
    import torchvision
    import os

    img_tensor = torch.clamp(img_tensor * ds + dm, 0, 1)
    if is_batch:
       for  i, img in enumerate(img_tensor):
            img = torchvision.transforms.ToPILImage()(img)
            img.save(path)
    else:
        # img.mul_(ds).add_(dm).clamp_(0, 1)
        # img = torchvision.transforms.ToPILImage()(img_tensor)
        # img.save(path)
        raise Exception('Except batch dimension in img tensor') 
def add_nosie_to_candicate(candidate, iteration, interval=100, num_levels=10, scale=0.1, start_iter=2000):
    """
    : candidate original sample
    : iterations current iteration
    """
    betas = reversed(torch.linspace(0, 10, num_levels) * scale)
    if iteration % interval == 0 and iteration > start_iter:
        idx = iteration // interval - 3

        if idx < num_levels:
            print(f'{iteration} add noise idx {idx}')
            noise = torch.randn_like(candidate).to(candidate.device)
            candidate.add_(noise * betas[idx])  
    

def get_location(val, length):
    if val < 0:
        return 0
    elif val > length:
        return length
    else:
        return val
    
#返回图像差异融合图片
def diff_imgs(img_batch, threshold=0.9):

    diff_list = []
    img_batch = img_batch.permute(0, 2, 3, 1).numpy()
    for i in range(img_batch.shape[0]):
        for j in range(i+1, img_batch.shape[0]):
            img1 = img_batch[i]
            img2 = img_batch[j]
            img1 = Image.fromarray(img1.astype('uint8')).convert('RGB')
            img2 = Image.fromarray(img2.astype('uint8')).convert('RGB')
            diff = ImageChops.difference(img1, img2)
            diff_list.append(diff)
    #merge multiple images
    merge_img = diff_list[0]
    for img in diff_list[1:]:
        merge_img = ImageChops.add(merge_img, img, scale=1, offset=0)
    return merge_img

#将图像中像素值较小的百分之20的像素点变为0，其他的变为1
def get_threshold(image, percent=20):
    #获取图像的直方图，返回值为元组，第一个元素为灰度值，第二个元素为该灰度值的像素数
    his = image.histogram()
    #像素总数
    total = sum(his)
    #累计像素数
    temp = 0
    #获取percent%的像素数对应的灰度值
    for i, h in enumerate(his):
        temp += h
        if temp > total * percent / 100:
            return i

def covert_to_binary(img, percent=40):
    img = img.convert('L')
    threshold = get_threshold(img, percent)
    img = img.point(lambda x: 0 if x < threshold else 255)
    return img

# print(add_1_2_3.size)
# covert_to_binary(merge_img.convert('L'), percent=40)


def get_mask(img_batch, percent=40):
    diff_img = diff_imgs(img_batch=img_batch)
    mask = covert_to_binary(diff_img, percent=percent)
    return mask
