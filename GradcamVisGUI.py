# __author__ = 'ChienHung Chen in Academia Sinica IIS'

from tkinter import *
import os, sys
from PIL import Image, ImageTk
import json
import cv2
import pickle
import numpy as np
import matplotlib
from tkinter import ttk
import xml.etree.ElementTree as ET
import argparse
import itertools

matplotlib.use("TkAgg")

# ==========================================================
# gradcam code
# this part is revised based on https://github.com/1Konny/gradcam_plus_plus-pytorch
# ==========================================================

import PIL
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

#from utils import visualize_cam, Normalize
#from gradcam import GradCAM, GradCAMpp

def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    
    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result


def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]
                
        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_vgg_layer(arch, target_layer_name):
    """Find vgg layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_42'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_alexnet_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_squeezenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features_12'
            target_layer_name = 'features_12_expand3x3'
            target_layer_name = 'features_12_expand3x3_activation'
            
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2]+'_'+hierarchy[3]]

    return target_layer



def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)
    
    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)
    
    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


import torch
import torch.nn.functional as F
import pdb

class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input, class_idx=None, channel_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)                   # [1, 1000]  (classified layer，no softmax)

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze() # max value
        else:
            score = logit[:, class_idx].squeeze()        # selected value
        
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)        # retain all mid graph, ensure they are not released

        if channel_idx == 'All':
            # show the summation of all channels
            gradients = self.gradients['value']              # [1, 512, 7, 7] for resnet18
            activations = self.activations['value']          # [1, 512, 7, 7] for resnet18
            b, k, u, v = gradients.size()            
            
        else:
            # show a single channel
            channel_idx = int(channel_idx)
            gradients = self.gradients['value'][:, channel_idx:(channel_idx+1), :, :]              # [1, 512, 7, 7] for resnet18
            activations = self.activations['value'][:, channel_idx:(channel_idx+1), :, :]          # [1, 512, 7, 7] for resnet18
            b, k, u, v = gradients.size()            

        alpha = gradients.view(b, k, -1).mean(2)         # [1, 512]
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)                 # [1, 512, 1, 1]

        saliency_map = (weights*activations).sum(1, keepdim=True)    # [1, 1, 7, 7]
        saliency_map = F.relu(saliency_map)
        saliency_map = torch.nn.functional.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data            

        return saliency_map, logit

    def __call__(self, input, class_idx=None, channel_idx=None, retain_graph=False):
        return self.forward(input, class_idx, channel_idx, retain_graph)


class GradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, channel_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze() 
            
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = gradients.size()

        if channel_idx == 'All':
            # show all channel summation
            gradients = self.gradients['value'] # dS/dA
            activations = self.activations['value'] # A
            b, k, u, v = gradients.size()
            
        else:
            # show single channel
            channel_idx = int(channel_idx)
            gradients = self.gradients['value'][:, channel_idx:(channel_idx+1), :, :]       # [1, 512, 7, 7] for resnet18
            activations = self.activations['value'][:, channel_idx:(channel_idx+1), :, :]   # [1, 512, 7, 7] for resnet18
            b, k, u, v = gradients.size() 

        # the difference between gradcam++ and gradcam 
        ############################################################
        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)
        ############################################################

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = torch.nn.functional.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit

# ==========================================================

parser = argparse.ArgumentParser(description="DetVisGUI")

# dataset information
parser.add_argument('--img_root', default='images', help='data image path')
parser.add_argument('--output', default='output', help='data image path')

args = parser.parse_args()

# ==========================================================


class dataset:
    def __init__(self):
        self.img_root = args.img_root
        self.img_list = sorted(os.listdir(self.img_root))

        # self.aug_category = aug_category(list(range(0,1000)))
        self.aug_category = aug_category(self.get_category_name())
        self.model_list = ['resnet18', 'resnet50', 'resnet101', 'vgg16', 'alexnet', 'densenet161', 'squeezenet1_1']

    def get_img_by_name(self, name):
        img = Image.open(os.path.join(self.img_root, name)).convert('RGB')
        return img

    def get_img_by_index(self, idx):
        img = Image.open(os.path.join(self.img_root, self.img_list[idx])).convert('RGB')
        return img

    def get_singleImg_gt(self, name):
        if name.replace('.jpg', '') not in self.total_annotations.keys():
            print('There are no annotations in %s.' % name)
            return []
        else:
            return self.total_annotations[name.replace('.jpg', '')]

    def get_singleImg_dets(self, name):
        return self.img_det[name]

    def get_category_name(self):
        with open('cetogory.txt' , 'r') as f: 
            data = f.readlines() 
        data = [d[:-1] for d in data]
        return data


# main GUI
class vis_tool:
    def __init__(self):
        self.window = Tk()
        self.menubar = Menu(self.window)

        self.info = StringVar()
        self.info_label = Label(self.window, bg='yellow', width=4, textvariable=self.info)

        self.listBox1 = Listbox(self.window, width=40, height=30, font=('Times New Roman', 10))
        self.scrollbar1 = Scrollbar(self.window, width=15, orient="vertical")

        self.listBox1_info = StringVar()
        self.listBox1_label = Label(self.window, font=('Arial', 11), bg='yellow', width=4, height=1, textvariable=self.listBox1_info)

        self.data_info = dataset()

        # load image and show it on the window
        self.img = self.data_info.get_img_by_index(0)
        self.photo = ImageTk.PhotoImage(self.img)
        self.label_img = Label(self.window, image=self.photo)

        self.combo_label = Label(self.window, bg='yellow', width=10, height=1, text='Show Category', font=('Arial', 11))
        self.combo_category = ttk.Combobox(self.window, font=('Arial', 11), values=self.data_info.aug_category.combo_list)
        self.combo_category.current(0)

        self.combo_label3 = Label(self.window, bg='yellow', width=10, height=1, text='Model', font=('Arial', 11))
        self.combo_category3 = ttk.Combobox(self.window, font=('Arial', 11), values=self.data_info.model_list)
        self.combo_category3.current(0)

        self.find_label = Label(self.window, font=('Arial', 11), bg='yellow', width=10, height=1, text="find")
        self.find_name = ""
        self.find_entry = Entry(self.window, font=('Arial', 11), textvariable=StringVar(self.window, value=str(self.find_name)), width=10)
        self.find_button = Button(self.window, text='Enter', height=1, command=self.findname)

        self.listBox1_idx = 0  # image listBox

        # ====== ohter attribute ======
        self.img_name = ''
        self.show_img = None
        self.output = args.output

        if not os.path.isdir(self.output):
            os.makedirs(self.output)

        self.img_list = self.data_info.img_list

        # flag for find/threshold button switch focused element
        self.button_clicked = False

        # ==============================
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        print(self.combo_category3.get())
        self.model = getattr(models, self.combo_category3.get())(pretrained=True)
        # print(self.model)
        
        self.model.eval() 
        self.model.cuda()

        print('====================================')
        layers = []
        self.channel_nums = {}

        for k, v in self.model.named_parameters():
            if '.bias' in k or len(v.shape) == 2:       
                continue
            new_k = k.replace('.weight', '')
            new_k = new_k.replace('.', '_')
            print('{:40}  {}'.format(k, v.shape))
            layers.append(new_k)
            self.channel_nums[new_k] = v.shape[0]
        
        print('====================================')
        self.data_info.layer_list = layers

        self.combo_label2 = Label(self.window, bg='yellow', width=10, height=1, text='Layer', font=('Arial', 11))
        self.combo_category2 = ttk.Combobox(self.window, font=('Arial', 11), values=self.data_info.layer_list)
        self.combo_category2.current(len(self.data_info.layer_list) - 1)

        self.combo_label4 = Label(self.window, bg='yellow', width=10, height=1, text='Channel', font=('Arial', 11))
        self.combo_category4 = ttk.Combobox(self.window, font=('Arial', 11), values=['All'] + list(range(self.channel_nums[self.combo_category2.get()])))
        self.combo_category4.current(0)


    def change_img(self, event=None):
        if len(self.listBox1.curselection()) != 0:
            self.listBox1_idx = self.listBox1.curselection()[0]

        self.listBox1_info.set("Image  {:6}  / {:6}".format(self.listBox1_idx + 1, self.listBox1.size()))

        name = self.listBox1.get(self.listBox1_idx)

        img = self.data_info.get_img_by_name(name)
        img, max_class_idx = self.run_gradcam(img)

        self.window.title(name + '   ' + self.data_info.aug_category.category[max_class_idx])
        
        self.img_name = name
        self.img = img
        self.show_img = img
        img = self.scale_img(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)
        self.window.update_idletasks()

        if self.img_name in os.listdir(self.output):
            self.listBox1_label.config(bg='#CCFF99')
        else:
            self.listBox1_label.config(bg='yellow')

    # ===============================================================

    def run_gradcam(self, pil_img):

        torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
        # torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
        torch_img = torch.nn.functional.interpolate(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
        normed_torch_img = self.normalizer(torch_img)

        cam_dict = dict()
        
        if 'vgg' in self.combo_category3.get():
            t = 'vgg'
        elif 'resnet' in self.combo_category3.get():
            t = 'resnet'
        elif 'densenet' in self.combo_category3.get():
            t = 'densenet'
        elif 'alexnet' in self.combo_category3.get():
            t = 'alexnet'
        elif 'squeezenet' in self.combo_category3.get():
            t = 'squeezenet'
        
        resnet_model_dict = dict(type=t, arch=self.model, layer_name=self.combo_category2.get(), input_size=(224, 224))
        resnet_gradcam = GradCAM(resnet_model_dict, verbose=False)
        resnet_gradcampp = GradCAMpp(resnet_model_dict, verbose=False)
        cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]


        images = []
        class_idx = None if self.combo_category.get() == 'Max' else int(self.combo_category.get().split(':')[0])

        for gradcam, gradcam_pp in cam_dict.values():
            mask, logit = gradcam(normed_torch_img, class_idx=class_idx, channel_idx=self.combo_category4.get())
            heatmap, result = visualize_cam(mask.cpu(), torch_img)
            
            mask_pp, logit = gradcam_pp(normed_torch_img, class_idx=class_idx, channel_idx=self.combo_category4.get())
            heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_img)
            
            # images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
            tmp = torch.cat([torch_img.squeeze().cpu(), heatmap, heatmap_pp], 2)
            tmp2 = torch.cat([torch_img.squeeze().cpu(), result, result_pp], 2)
            images.append(torch.cat([tmp, tmp2], 1))

            max_class_idx = logit.max(1)[-1]
        
        # images = make_grid(torch.cat(images, 0), nrow=5)
        images = make_grid(torch.cat(images, 0), nrow=3)
        images = images.numpy().transpose(1,2,0)
        images = Image.fromarray((images * 255).astype(np.uint8))

        return images, max_class_idx


    def scale_img(self, img):
        [s_w, s_h] = [1, 1]
        #pdb.set_trace()
        # if window size is (1920, 1080), the default max image size is (1440, 810)
        (fix_width, fix_height) = (1440, 810)

        # change image size according to window size
        if self.window.winfo_width() != 1:
            fix_width = (self.window.winfo_width() - self.listBox1.winfo_width() - self.scrollbar1.winfo_width() - 5)
            fix_height = int(fix_width * 9 / 16)

        # handle image size is too big
        if img.width > fix_width:
            s_w = fix_width / img.width
        if img.height > fix_height:
            s_h = fix_height / img.height

        scale = min(s_w, s_h)

        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.ANTIALIAS)
        return img

    def save_img(self):
        print('Save Image to ' + os.path.join(self.output, self.img_name))
        cv2.imwrite(os.path.join(self.output, self.img_name), cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB))
        self.listBox1_label.config(bg='#CCFF99')


    def eventhandler(self, event):
        if self.window.focus_get() not in [self.find_entry]:
            if event.keysym == 'q':
                self.window.quit()
            if event.keysym == 's':
                self.save_img()

            if self.button_clicked:
                self.button_clicked = False


    def combobox_change(self, event=None):
        self.listBox1.focus()
        self.change_img()


    def layer_change(self, event=None):
        self.combo_category4['values'] = ['All'] + list(range(self.channel_nums[self.combo_category2.get()]))
        self.combo_category4.current(0)

        self.listBox1.focus()
        self.change_img()


    def model_change(self, event=None):
        del self.model
        
        print(self.combo_category3.get())
        self.model = getattr(models, self.combo_category3.get())(pretrained=True)
        print(self.model)
        
        self.model.eval() 
        self.model.cuda()

        print('====================================')
        layers = []
        self.channel_nums = {}
        # pdb.set_trace()
        for k, v in self.model.named_parameters():
            if '.bias' in k or len(v.shape) == 2:      
                continue
            new_k = k.replace('.weight', '')
            new_k = new_k.replace('.', '_')
            print('{:40}  {}'.format(k, v.shape))
            layers.append(new_k)
            self.channel_nums[new_k] = v.shape[0]
        
        print('====================================')

        self.data_info.layer_list = layers        
        self.combo_category2['values'] = self.data_info.layer_list
        self.combo_category2.current(len(self.data_info.layer_list) - 1)

        self.combo_category4['values'] = ['All'] + list(range(self.channel_nums[self.combo_category2.get()]))
        self.combo_category4.current(0)

        self.listBox1.focus()
        self.change_img()


    def clear_add_listBox1(self):
        self.listBox1.delete(0, 'end')  # delete listBox1 0 ~ end items

        # add image name to listBox1
        for item in self.img_list:
            self.listBox1.insert('end', item)

        self.listBox1.select_set(0)
        self.listBox1.focus()
        self.change_img()


    def findname(self, event=None):
        self.find_name = self.find_entry.get()
        new_list = []

        if self.find_name == '':
            new_list = self.data_info.img_list
        else:
            for img_name in self.data_info.img_list:
                if self.find_name[0] == "!":
                    if self.find_name[1:] not in img_name:
                        new_list.append(img_name)
                else:
                    if self.find_name in img_name:
                        new_list.append(img_name)

        if len(new_list) != 0:
            self.img_list = new_list
            self.clear_add_listBox1()
            # self.clear_add_listBox2()
            self.button_clicked = True
        else:
            self.window.title("Can't find any image about '{}'".format(self.find_name))


    def run(self):
        self.window.geometry('1280x800+350+100')

        # self.menubar.add_command(label='QUIT', command=self.window.quit)
        # self.window.config(menu=self.menubar)                               # display the menu
        self.scrollbar1.config(command=self.listBox1.yview)
        self.listBox1.config(yscrollcommand=self.scrollbar1.set)

        layer1 = 0
        layer2 = 50

        # ======================= layer 1 =========================

        # combobox
        self.combo_label.grid(row=layer1 + 30, column=0, sticky=W + E + N + S, padx=3, pady=3, columnspan=6)
        self.combo_category.grid(row=layer1 + 30, column=6, sticky=W + E + N + S, padx=3, pady=3, columnspan=21)

        self.combo_label3.grid(row=layer1 + 40, column=0, sticky=W + E + N + S, padx=3, pady=3, columnspan=6)
        self.combo_category3.grid(row=layer1 + 40, column=6, sticky=W + E + N + S, padx=3, pady=3, columnspan=6)

        self.combo_label2.grid(row=layer1 + 40, column=12, sticky=W + E + N + S, padx=3, pady=3, columnspan=3)
        self.combo_category2.grid(row=layer1 + 40, column=15, sticky=W + E + N + S, padx=3, pady=3, columnspan=6)

        self.combo_label4.grid(row=layer1 + 40, column=21, sticky=W + E + N + S, padx=3, pady=3, columnspan=3)
        self.combo_category4.grid(row=layer1 + 40, column=24, sticky=W + E + N + S, padx=3, pady=3, columnspan=3)

        # ======================= layer 2 =========================

        self.listBox1_label.grid(row=layer2 + 0, column=0, sticky=N + S + E + W, columnspan=12)

        # find name
        self.find_label.grid(row=layer2 + 20, column=0, sticky=E + W, columnspan=4)
        self.find_entry.grid(row=layer2 + 20, column=4, sticky=E + W, columnspan=4)
        self.find_button.grid(row=layer2 + 20, column=8, sticky=E + W, pady=3, columnspan=4)

        self.scrollbar1.grid(row=layer2 + 30, column=11, sticky=N + S + W)
        self.label_img.grid(row=layer2, column=12, sticky=N + E, padx=3, rowspan=110, columnspan=15)
        self.listBox1.grid(row=layer2 + 30, column=0, sticky=N + S + E + W, pady=3, columnspan=11)

        self.clear_add_listBox1()
        self.listBox1.bind('<<ListboxSelect>>', self.change_img)
        self.listBox1.bind_all('<KeyRelease>', self.eventhandler)

        self.find_entry.bind('<Return>', self.findname)
        self.find_entry.bind('<KP_Enter>', self.findname)

        self.combo_category.bind("<<ComboboxSelected>>", self.combobox_change)
        self.combo_category2.bind("<<ComboboxSelected>>", self.layer_change)
        self.combo_category3.bind("<<ComboboxSelected>>", self.model_change)
        self.combo_category4.bind("<<ComboboxSelected>>", self.combobox_change)

        self.window.mainloop()


class aug_category:
    def __init__(self, categories):
        self.category = categories
        self.combo_list = categories.copy()
        self.combo_list.insert(0, 'Max')
        self.all = True


if __name__ == "__main__":
    vis_tool().run()


# class name
# https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
# https://blog.csdn.net/LegenDavid/article/details/73335578

