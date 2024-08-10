import os
from PIL import Image
from torch.utils.data import Dataset
import math
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import argparse
import shutil
from datetime import datetime
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-device", "--device", help="The id of the GPU", default=f"cuda:0")
parser.add_argument("-video_index_base", "--video_index_base", help="The start index of encoding video", type = int, default=0)
parser.add_argument("-encode_video_num", "--encode_video_num", help="The encoded video num", type = int, default=10)
parser.add_argument("-B_INR_NeRV_weight_path", "--B_INR_NeRV_weight_path", help="The background INR weight save path", default=f"./B_INR_weights")
parser.add_argument("-epochs", "--epochs", help="The NeRV training epochs", type = int, default=300)
parser.add_argument("-temporal_embed", "--temporal_embed", help="The temporal embedding", default=f"1.25_40")
parser.add_argument("-stem_dim_num", "--stem_dim_num", help="The MLP hidden layer dim", default=f"256_1")
parser.add_argument("-fc_hw_dim", "--fc_hw_dim", help="fc_hw_dim", default=f"9_16_16")
parser.add_argument("-train_image_path", "--train_image_path", help="The video frame path for NeRV encoding", default=f"../data/OTB")
parser.add_argument("-lower_width", "--lower_width", help="lower_width", type = int, default=96)

args = parser.parse_args()

epochs = args.epochs
warmup = 0.2
warmup = int (epochs * warmup)
manualSeed = 1
embed = args.temporal_embed # 1.25 is the value of b and 40 is the length of the temporal embedding
stem_dim_num = args.stem_dim_num
fc_hw_dim = args.fc_hw_dim
expansion = 1
num_blocks= 1
norm = 'none'
act = 'swish'
reduction = 2
conv_type = 'conv'
strides = [5, 2, 2, 2]
single_res = True
lower_width = args.lower_width
sigmoid = True
loss_type = 'Fusion6'
lr_raw = 0.0005
eval_freq = 50
device = args.device

split_num = 1 # here we choose to encode the whole dataset together
beta = 0.5
video_index_base = args.video_index_base
encode_video_num = args.encode_video_num
train_list_path = args.train_image_path
val_list_path = train_list_path


B_INR_NeRV_weight_path = args.B_INR_NeRV_weight_path


def print_model_size_in_mb(model):
    """
    Print the size of a PyTorch model in kilobytes.
    
    Args:
    model (torch.nn.Module): The PyTorch model.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    total_size = param_size / 1024/ 1024
    print(f"Model Size: {total_size:.2f} MB")



# customized Dataset object used for NeRV dataloader
def is_image_file(filename): # Compares 'filename' extension to common image file types.
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
def load_image_path(imgDir):

    all_training_files=os.walk(imgDir)
    train_files=[]
    train_imageNames=[]
    train_nSamples=0
    for path,direction,filelist in all_training_files:
        files = [file for file in filelist if os.path.isfile(os.path.join(path, file))]
        imageNames = [file.split('.')[0] for file in files if is_image_file(file)]
        files = [os.path.join(path, file) for file in files if is_image_file(file)]
        train_files.append(files)
        train_imageNames.append(imageNames)
        train_nSamples=train_nSamples+len(files)
    train_files=sum(train_files,[])
    train_imageNames=sum(train_imageNames,[])
    print(train_imageNames[0])
    print(train_files[0])
    
    return train_files, train_imageNames

def get_all_folder_paths(directory):
    file_paths = []
    dir_list = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            dir_list.append(folder_path)
    dir_list.sort()
    return dir_list


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, vid_list=[None], frame_gap=1,  visualize=False, split_num = 1, choose_part = 0):
        self.main_dir = main_dir
        self.transform = transform
        frame_idx, self.frame_path = [], []
        accum_img_num = []
        all_imgs = os.listdir(main_dir)
        all_imgs.sort()
        #print("the length of all_imgs is:", len(all_imgs))
        img_seg_num = int(len(all_imgs)/split_num)
        # img_remain_num = len(all_imgs)%split_num
        # img_seg_list_new = []
        if (choose_part != split_num - 1):
            img_seg_list_new = all_imgs[int(choose_part*img_seg_num):int((choose_part+1)*img_seg_num)]
        else:
            img_seg_list_new = all_imgs[int(choose_part*img_seg_num):len(all_imgs) + 1]

        num_frame = 0 
        for img_id in img_seg_list_new:
            self.frame_path.append(img_id)
            frame_idx.append(num_frame)
            num_frame += 1          

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        accum_img_num.append(num_frame)
        self.frame_idx = [float(x) / len(frame_idx) for x in frame_idx]
        self.accum_img_num = np.asfarray(accum_img_num)
        if None not in vid_list:
            self.frame_idx = [self.frame_idx[i] for i in vid_list]
        self.frame_gap = frame_gap

    def __len__(self):
        return len(self.frame_idx) // self.frame_gap

    def __getitem__(self, idx):
        valid_idx = idx * self.frame_gap
        img_id = self.frame_path[valid_idx]
        img_name = os.path.join(self.main_dir, img_id)
        image = Image.open(img_name).convert("RGB")
        image = image.resize((640, 360))
        tensor_image = self.transform(image)
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0,2,1)
        frame_idx = torch.tensor(self.frame_idx[valid_idx])

        return tensor_image, frame_idx
    
# some basic class needed by NeRV

class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class CustomConv(nn.Module):
    def __init__(self, **kargs):
        super(CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        out = self.conv(x)
        #print("the shape of the conv is:", out.shape)
        temp = self.up_scale(out)
        #print("the shape of the upscaled is:", temp.shape)
        return self.up_scale(out)


def MLP(dim_list, act='relu', bias=True):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.conv = CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], stride=kargs['stride'], bias=kargs['bias'], 
            conv_type=kargs['conv_type'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
        mlp_dim_list = [kargs['embed_length']] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]
        self.stem = MLP(dim_list=mlp_dim_list, act=kargs['act'])
        
        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        for i, stride in enumerate(kargs['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * kargs['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])

            for j in range(kargs['num_blocks']):
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                    bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'], conv_type=kargs['conv_type']))
                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if kargs['sin_res']:
                if i == len(kargs['stride_list']) - 1:
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias']) 
                    # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias']) 
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias'])
                # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias'])
            self.head_layers.append(head_layer)
        self.sigmoid =kargs['sigmoid']

    def forward(self, input):
        output = self.stem(input)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)
        #print("the output shape of MLP is:", output.shape)

        out_list = []
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output) 
            #print("the shape of the output is:", output.shape)
            if head_layer is not None:
                img_out = head_layer(output)
                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)

        return  out_list
    
# some other useful functions

def quantize_per_tensor(t, bit=8, axis=-1):
    if axis == -1:
        t_valid = t!=0
        t_min, t_max =  t[t_valid].min(), t[t_valid].max()
        scale = (t_max - t_min) / 2**bit
    elif axis == 0:
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i]!=0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)        
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[:,None,None,None]
            t_min = min_max_tf[:,0,None,None,None]
        elif t.dim() == 2:
            scale = scale[:,None]
            t_min = min_max_tf[:,0,None]
    elif axis == 1:
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:,i]!=0
            if t_valid.sum():
                min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)             
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[None,:,None,None]
            t_min = min_max_tf[None,:,0,None,None]
        elif t.dim() == 2:
            scale = scale[None,:]
            t_min = min_max_tf[None,:,0]            
    # import pdb; pdb.set_trace; from IPython import embed; embed()       
    quant_t = ((t - t_min) / (scale + 1e-19)).round()
    new_t = t_min + scale * quant_t
    return quant_t, new_t
    
def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionalEncoding, self).__init__()
        self.pe_embed = pe_embed.lower()
        if self.pe_embed == 'none':
            self.embed_length = 1
        else:
            self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels

    def forward(self, pos):
        if self.pe_embed == 'none':
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1)


def psnr2(img1, img2):
    mse = (img1 - img2) ** 2
    PIXEL_MAX = 1
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    return psnr

def loss_fn(pred, target, loss_type):
    target = target.detach()

    if loss_type == 'L2':
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.mean()       
    elif loss_type == 'L1':
        loss = torch.mean(torch.abs(pred - target))
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = 0.3 * F.mse_loss(pred, target) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = 0.3 * torch.mean(torch.abs(pred - target)) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion4':
        loss = 0.5 * torch.mean(torch.abs(pred - target)) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion5':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion6':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion7':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * torch.mean(torch.abs(pred - target))
    elif loss_type == 'Fusion8':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * torch.mean(torch.abs(pred - target))
    elif loss_type == 'Fusion9':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion10':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion11':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion12':
        loss = 0.8 * torch.mean(torch.abs(pred - target)) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    return loss

def psnr_fn(output_list, target_list):
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction='mean')
        psnr = -10 * torch.log10(l2_loss)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1) #(batchsize, num_stage)
    return psnr

def msssim_fn(output_list, target_list):
    msssim_list = []
    for output, target in zip(output_list, target_list):
        if output.size(-2) >= 160:
            msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0) #(num_stage)
    msssim = msssim.view(1, -1).expand(output_list[-1].size(0), -1) #(batchsize, num_stage)
    return msssim

def RoundTensor(x, num=2, group_str=False):
    if group_str:
        str_list = []
        for i in range(x.size(0)):
            x_row =  [str(round(ele, num)) for ele in x[i].tolist()]
            str_list.append(','.join(x_row))
        out_str = '/'.join(str_list)
    else:
        str_list = [str(round(ele, num)) for ele in x.flatten().tolist()]
        out_str = ','.join(str_list)
    return out_str

def adjust_lr(optimizer, cur_epoch, cur_iter, data_size, lr_type, warmup, lr, epochs, lr_steps):
    cur_epoch = cur_epoch + (float(cur_iter) / data_size)
    if lr_type == 'cosine':
        lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - warmup)/ (epochs - warmup)) + 1.0)
    elif lr_type == 'step':
        lr_mult = 0.1 ** (sum(cur_epoch >= np.array(lr_steps)))
    elif lr_type == 'const':
        lr_mult = 1
    elif lr_type == 'plateau':
        lr_mult = 1
    else:
        raise NotImplementedError

    if cur_epoch < warmup:
        lr_mult = 0.1 + 0.9 * cur_epoch / warmup

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * lr_mult

    return lr * lr_mult

def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

class PositionalEncodingTrans(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        index = torch.round(pos * self.max_len).long()
        p = self.pe[index]
        return p
    


cudnn.benchmark = True
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)

is_train_best, is_val_best = False, False
PE = PositionalEncoding(embed)
embed_length = PE.embed_length
model = Generator(embed_length=embed_length, stem_dim_num=stem_dim_num, fc_hw_dim=fc_hw_dim, expansion=expansion, 
    num_blocks=num_blocks, norm=norm, act=act, bias = True, reduction=reduction, conv_type=conv_type,
    stride_list=strides,  sin_res=single_res,  lower_width=lower_width, sigmoid=sigmoid)

print_model_size_in_mb(model)
print(model)
   
local_rank = None
##### get model params and flops #####
total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
if local_rank in [0, None]:
    params = sum([p.data.nelement() for p in model.parameters()]) / 1e6

    print(f' {model}\n Model Params: {params}M')
    # with open('{}/rank0.txt'.format(args.outf), 'a') as f:
    #     f.write(str(model) + '\n' + f'Params: {params}M\n')
    # writer = SummaryWriter(os.path.join(args.outf, f'param_{total_params}M', 'tensorboard'))
else:
    writer = None
    
@torch.no_grad()
def evaluate(model, val_dataloader, pe, local_rank, quant_bit, quant_axis, dump_images, debug, eval_fps, batchSize, print_freq, device):
    # Model Quantization
    if quant_bit != -1:
        cur_ckt = model.state_dict()
        from dahuffman import HuffmanCodec
        quant_weitht_list = []
        for k,v in cur_ckt.items():
            large_tf = (v.dim() in {2,4} and 'bias' not in k)
            quant_v, new_v = quantize_per_tensor(v, quant_bit, quant_axis if large_tf else -1)
            valid_quant_v = quant_v[v!=0] # only include non-zero weights
            quant_weitht_list.append(valid_quant_v.flatten())
            cur_ckt[k] = new_v
        cat_param = torch.cat(quant_weitht_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(input_code_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        avg_bits = total_bits / len(input_code_list)    
        # import pdb; pdb.set_trace; from IPython import embed; embed()       
        encoding_efficiency = avg_bits / quant_bit
        print_str = f'Entropy encoding efficiency for bit {quant_bit}: {encoding_efficiency}'
        print(print_str)    
        model.load_state_dict(cur_ckt)

        # import pdb; pdb.set_trace; from IPython import embed; embed()

    psnr_list = []
    msssim_list = []

    time_list = []
    model.eval()
    for i, (data,  norm_idx) in enumerate(val_dataloader):
        if i > 10 and debug:
            break
        embed_input = pe(norm_idx)
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            embed_input = embed_input.cuda(local_rank, non_blocking=True)
        else:
            #data,  embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)
            data, embed_input = data.to(device), embed_input.to(device)

        # compute psnr and msssim
        fwd_num = 10 if eval_fps else 1
        for _ in range(fwd_num):
            # embed_input = embed_input.half()
            # model = model.half()
            start_time = datetime.now()
            output_list = model(embed_input)
            torch.cuda.synchronize()
            # torch.cuda.current_stream().synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        psnr_list.append(psnr_fn(output_list, target_list))
        msssim_list.append(msssim_fn(output_list, target_list))
        val_psnr = torch.cat(psnr_list, dim=0)              #(batchsize, num_stage)
        val_psnr = torch.mean(val_psnr, dim=0)              #(num_stage)
        val_msssim = torch.cat(msssim_list, dim=0)          #(batchsize, num_stage)
        val_msssim = torch.mean(val_msssim.float(), dim=0)  #(num_stage)        
        if i % print_freq == 0:
            fps = fwd_num * (i+1) * batchSize / sum(time_list)
            print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {}'.format(
                local_rank, i+1, len(val_dataloader),
                RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False), round(fps, 2))
            print(print_str)
    model.train()

    return val_psnr, val_msssim


checkpoint = None
start_epoch = 0
not_resume_epoch = True
img_transforms = transforms.ToTensor()
DataSet = CustomDataSet


train_data_dir_array = get_all_folder_paths(train_list_path)
val_data_dir_array = get_all_folder_paths(val_list_path)

train_data_dir = list(train_data_dir_array)
val_data_dir = list(val_data_dir_array)

    
workers = 8
vid = [None]
frame_gap = 1
test_gap = 1
eval_only = False
batchSize = 1
video_length = len(train_data_dir)




for video_index in range (encode_video_num):
        train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]
        
        for chunk_idx in range (1):
                #split_num_current = int(split_num[video_index_base + video_index])  
                temp_list = train_data_dir[video_index_base + video_index].split('/')
                object_theme = temp_list[-1]
                print("the current object is:", object_theme)
                train_dataset = DataSet(str(train_data_dir[video_index_base + video_index]), img_transforms,vid_list=vid, frame_gap=frame_gap,  split_num = 1, choose_part = chunk_idx)
                #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
                train_sampler = None
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=(train_sampler is None),
                        num_workers=workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

                val_dataset = DataSet(str(val_data_dir[video_index_base + video_index]), img_transforms, vid_list=vid, frame_gap=test_gap,  split_num = 1, choose_part = chunk_idx)
                #val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if distributed else None
                val_sampler = None
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize,  shuffle=False,
                        num_workers=workers, pin_memory=True, sampler=val_sampler, drop_last=False, worker_init_fn=worker_init_fn)
                data_size = len(train_dataset)
                
                
                cycles = 1
                debug = False
                lw = 1.0
                lr_type = 'cosine'
                lr_steps = []
                print_freq = 50
                quant_bit = -1
                quant_axis = 0
                dump_images = False
                eval_fps = False
                test_gap = 1

                model_new = Generator(embed_length=embed_length, stem_dim_num=stem_dim_num, fc_hw_dim=fc_hw_dim, expansion=expansion, 
                        num_blocks=num_blocks, norm=norm, act=act, bias = True, reduction=reduction, conv_type=conv_type,
                        stride_list=strides,  sin_res=single_res,  lower_width=lower_width, sigmoid=sigmoid)
                
                model_new = model_new.to(device)
                
                optimizer = optim.Adam(model_new.parameters(), betas=(beta, 0.999))
                # Training
                start = datetime.now()
                total_epochs = epochs * cycles
                print(total_epochs)
                for epoch in range(start_epoch, total_epochs):
                        model_new.train()
                        
                        epoch_start_time = datetime.now()
                        psnr_list = []
                        msssim_list = []
                        # iterate over dataloader
                        for i, (data,  norm_idx) in enumerate(train_dataloader):
                                if i > 10 and debug:
                                        break
                                embed_input = PE(norm_idx)
                                if local_rank is not None:
                                        data = data.cuda(local_rank, non_blocking=True)
                                        embed_input = embed_input.cuda(local_rank, non_blocking=True)
                                else:
                                        #data,  embed_input = data.cuda(non_blocking=True),   embed_input.cuda(non_blocking=True)
                                        data, embed_input = data.to(device), embed_input.to(device)
                                        # forward and backward
                                        output_list = model_new(embed_input)
                        
                                target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
                                loss_list = [loss_fn(output, target, loss_type) for output, target in zip(output_list, target_list)]
                                loss_list = [loss_list[i] * (lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
                                loss_sum = sum(loss_list)
                                lr = adjust_lr(optimizer, epoch % epochs, i, data_size, lr_type, warmup, lr_raw, epochs, lr_steps )  
                                optimizer.zero_grad()
                                loss_sum.backward()
                                optimizer.step()

                                # compute psnr and msssim
                                psnr_list.append(psnr_fn(output_list, target_list))
                                msssim_list.append(msssim_fn(output_list, target_list))
                                if i % print_freq == 0 or i == len(train_dataloader) - 1:
                                        train_psnr = torch.cat(psnr_list, dim=0) #(batchsize, num_stage)
                                        train_psnr = torch.mean(train_psnr, dim=0) #(num_stage)
                                        train_msssim = torch.cat(msssim_list, dim=0) #(batchsize, num_stage)
                                        train_msssim = torch.mean(train_msssim.float(), dim=0) #(num_stage)
                                        time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                                        print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}'.format(
                                                time_now_string, local_rank, epoch+1, epochs, i+1, len(train_dataloader), lr, 
                                                RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False))
                                        print(print_str, flush=True)

                        # ADD train_PSNR TO TENSORBOARD
                        if local_rank in [0, None]:
                                h, w = output_list[-1].shape[-2:]
                                is_train_best = train_psnr[-1] > train_best_psnr
                                train_best_psnr = train_psnr[-1] if train_psnr[-1] > train_best_psnr else train_best_psnr
                                train_best_msssim = train_msssim[-1] if train_msssim[-1] > train_best_msssim else train_best_msssim
                                print_str = '\t{}p: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(h, train_psnr[-1].item(), train_best_psnr.item(), train_best_msssim.item())
                                print(print_str, flush=True)
                                epoch_end_time = datetime.now()
                                print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                                        (epoch_end_time - start).total_seconds() / (epoch + 1 - start_epoch) ))

                        state_dict = model_new.state_dict()
                        save_checkpoint = {
                                'epoch': epoch+1,
                                'state_dict': state_dict,
                                'train_best_psnr': train_best_psnr,
                                'train_best_msssim': train_best_msssim,
                                'val_best_psnr': val_best_psnr,
                                'val_best_msssim': val_best_msssim,
                                'optimizer': optimizer.state_dict(),   
                        }    
                        # evaluation
                        if (epoch + 1) % eval_freq == 0 or epoch > total_epochs - 10:
                                val_start_time = datetime.now()
                                #model, val_dataloader, pe, local_rank, quant_bit, quant_axis, dump_images, debug, eval_fps, batchSize, print_freq
                                val_psnr, val_msssim = evaluate(model_new, val_dataloader, PE, local_rank, quant_bit, quant_axis, dump_images, debug, eval_fps, batchSize, print_freq, device)
                                val_end_time = datetime.now()      
                                if local_rank in [0, None]:
                                # ADD val_PSNR TO TENSORBOARD
                                        h, w = output_list[-1].shape[-2:]
                                        print_str = f'Eval best_PSNR at epoch{epoch+1}:'
                                        is_val_best = val_psnr[-1] > val_best_psnr
                                        val_best_psnr = val_psnr[-1] if is_val_best else val_best_psnr
                                        val_best_msssim = val_msssim[-1] if val_msssim[-1] > val_best_msssim else val_best_msssim
                                        print_str += '\t{}p: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}\t Time/epoch: {:.2f}'.format(h, val_psnr[-1].item(),
                                                val_best_psnr.item(), val_best_msssim.item(), (val_end_time - val_start_time).total_seconds())
                                        print(print_str)
                                # with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                                #     f.write(print_str + '\n')
                                        if is_val_best:
                                                torch.save(save_checkpoint, B_INR_NeRV_weight_path + '/' + f'model_val_best_{object_theme}_{chunk_idx}.pth') 

                        if local_rank in [0, None]:
                                # state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                                torch.save(save_checkpoint, B_INR_NeRV_weight_path + '/' + f'model_latest_{object_theme}_{chunk_idx}.pth')
                                if is_train_best:
                                        torch.save(save_checkpoint, B_INR_NeRV_weight_path + '/' + f'model_train_best_{object_theme}_{chunk_idx}.pth')
