import numpy as np
import cv2
import time
import torch
from torch import nn
from math import sqrt
from typing import Dict
from torch._C import dtype
from torchvision import transforms
from collections import OrderedDict
import tqdm
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import pathlib
from types import FunctionType
from PIL import Image, ImageColor, ImageDraw, ImageFont
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-device", "--device", help="The id of the GPU", default=f"cuda:0")
parser.add_argument("-train_object_num", "--train_object_num", help="The number of trained objects", type = int, default=1)
parser.add_argument("-object_idx_offset", "--object_idx_offset", help="The object idx start", type = int, default=5)
parser.add_argument("-img_idx_offset", "--img_idx_offset", help="The img idx start", type = int, default=0)
parser.add_argument("-object_category_num", "--object_category_num", help="The number of object categories", type = int, default=95)
parser.add_argument("-training_epochs", "--training_epochs", help="The number of training epochs", type = int, default=10000)
parser.add_argument("-lr", "--lr", help="The learning rate for INR encoding", type = float, default=2e-4)
parser.add_argument("-w0", "--w0", help="The sine activation function frequency", type = float, default=30.0)
parser.add_argument("-w0_initial", "--w0_initial", help="The sine activation function initial frequency", type = float, default=30.0)


parser.add_argument("-object_INR_list", "--object_INR_list", help="The number of trained objects",
                    default="10,3,15,3,17,5,24,5",
                    type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument("-area_threshold", "--area_threshold", help="The area threshold to decide INR size",
                    default="1001, 3001, 10001",
                    type=lambda s: [int(item) for item in s.split(',')])

parser.add_argument("-area_array_path", "--area_array_path", help="The object area array path", default=f"../area_info")
parser.add_argument("-train_list_path", "--train_list_path", help="The object residual array", default=f"./residual_array")
parser.add_argument("-train_ref_list_path", "--train_ref_list_path", help="The raw object array", default=f"./raw_object_array")
parser.add_argument("-train_recon_list_path", "--train_recon_list_path", help="The INR reconstructed object array", default=f"./B_INR_recon_object_array")

parser.add_argument("-npy_image_save_path", "--npy_image_save_path", help="The path to save the reconstructured objects in npy format", default=f"./object_npy")
parser.add_argument("-object_INR_weight_save_path", "--object_INR_weight_save_path", help="The path to save the object INR weights", default=f"./O_INR_weights")
parser.add_argument("-png_image_save_path", "--png_image_save_path", help="The path to save the object png images", default=f"./object_image_png")

parser.add_argument("-log_dir_path", "--log_dir_path", help="The path to save the log files", default=f"./logdir_logfiles")
args = parser.parse_args()


w0 = args.w0
w0_initial = args.w0_initial
learning_rate = args.lr
num_iters = args.training_epochs
object_category_num = args.object_category_num
device = args.device
train_object_num = args.train_object_num
object_idx_offset = args.object_idx_offset
img_idx_offset = args.img_idx_offset
train_list_path = args.train_list_path
train_ref_list_path = args.train_ref_list_path
train_recon_list_path = args.train_recon_list_path

object_INR_list = args.object_INR_list
hidden_dim_class_1 = object_INR_list[0]
hidden_layer_class_1 = object_INR_list[1]
hidden_dim_class_2 = object_INR_list[2]
hidden_layer_class_2 = object_INR_list[3]
hidden_dim_class_3 = object_INR_list[4]
hidden_layer_class_3 = object_INR_list[5]
hidden_dim_class_4 = object_INR_list[6]
hidden_layer_class_4 = object_INR_list[7]

area_threshold_list = args.area_threshold
area_threshold_1 = area_threshold_list[0]
area_threshold_2 = area_threshold_list[1]
area_threshold_3 = area_threshold_list[2]


INR_encoded_weights_save_path = args.object_INR_weight_save_path
image_object_save_path = args.npy_image_save_path
area_array_path = args.area_array_path

train_list_path = args.train_list_path
train_ref_list_path = args.train_ref_list_path
train_recon_list_path = args.train_recon_list_path


logdir_logfiles = args.log_dir_path
if not os.path.exists(logdir_logfiles):
    os.makedirs(logdir_logfiles)


def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

   

def is_npy_file(filename): # Compares 'filename' extension to common image file types.
    return any(filename.endswith(extension) for extension in ['.npy'])
def load_image_path(imgDir):

    all_training_files=os.walk(imgDir)
    train_files=[]
    train_imageNames=[]
    train_nSamples=0
    for path,direction,filelist in all_training_files:
        files = [file for file in filelist if os.path.isfile(os.path.join(path, file))]
        imageNames = [file.split('.')[0] for file in files if is_npy_file(file)]
        files = [os.path.join(path, file) for file in files if is_npy_file(file)]
        train_files.append(files)
        train_imageNames.append(imageNames)
        train_nSamples=train_nSamples+len(files)
    train_files=sum(train_files,[])
    train_imageNames=sum(train_imageNames,[])
    #print(train_imageNames[0])
    #print(train_files[0])
    
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


DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}

def save_image_new(image, address):
    image = image * 255
    image = image.round()
    image = torch.clamp(image,0,255)
    image_array = image.cpu().numpy()
    image_array = image_array.transpose(1,2,0)
    cv2.imwrite(address, image_array) 

def save_image_PIL(image, address):
    image = image * 255
    image = image.round()
    image = torch.clamp(image,0,255)
    image_array = image.cpu().numpy()
    image_array = image_array.transpose(1,2,0)
    image_array = image_array.astype(np.uint8)
    pil_image = Image.fromarray(image_array)
    pil_image.save(address) 


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.
    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    #print("the shape of the coordinates is:", coordinates.shape)
    # Normalize coordinates to lie in [-.5, .5]
    coordinates[:,0] = coordinates[:,0]/ (img.shape[1] - 1) - 0.5
    coordinates[:,1] = coordinates[:,1]/ (img.shape[2] - 1) - 0.5
    #coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features

def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))

def bpp(image, model):
    """Computes size in bits per pixel of model.
    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels

def psnr(img1, img2):
    """Calculates PSNR between two images.
    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()

def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.
    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.

def get_clamped_psnr(img_recon, img):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.
    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return psnr(clamp_image(img_recon), img)

@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    

def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;
    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")


def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(make_grid)
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

train_image_num = 0
#dir_appendix = f'_{num_layers}x{layer_size}_{w0:.0f}fq_{num_iters}ep_{train_image_num}im'
dir_appendix = f'{w0:.0f}fq_{num_iters}ep_{train_image_num}im'
logfile = open(logdir_logfiles + '/logfile' + dir_appendix + '.txt','w')
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

train_times = []
avg_psnr = 0
total_train_time = 0
avg_train_time = 0
model_size = 0
logfile.write('image_num total_inr_fitting_time best_training_psnr\n')

class Sine(nn.Module):
    """Sine activation with scaling.
    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.
    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool):
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out

class Siren(nn.Module):
    """SIREN model.
    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(SirenLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        self.net = nn.Sequential(*layers)

        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = SirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0,
                                use_bias=use_bias, activation=final_activation)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)
    

class Trainer():
    def __init__(self, representation, lr=1e-3, print_freq=1):
        """Model to learn a representation of a single datapoint.
        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_iters)
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())
        self.current_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())

    def train(self, coordinates, features, num_iters, img, Q, img_recon, img_raw):
        """Fit neural net to image.
        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        #with tqdm.trange(num_iters, ncols=100) as t:
        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
        #for i in range(num_iters):
            # Update model
                self.optimizer.zero_grad()
                predicted = self.representation(coordinates)
                loss = self.loss_func(predicted, features)
                loss.backward()
                self.optimizer.step()
                reconstructured_features = predicted * Q
                reversed_img = reconstructured_features.T.reshape(img.shape)
                reversed_image_tran = reversed_img.permute(2,1,0)
                img_combined = img_recon - reversed_image_tran
                img_raw = img_raw
                img_combined = img_combined/255.0

                # Calculate psnr
                psnr = get_clamped_psnr(img_combined, img_raw)

                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                for k, v in self.representation.state_dict().items():
                            self.current_model[k].copy_(v)
                
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)
                self.scheduler.step()


# train_list_path = '/export/hdd/scratch/dataset/DAC-SDC/cropped_object_residual/'  # object residual array
# train_ref_list_path = '/export/hdd/scratch/dataset/DAC-SDC/cropped_raw_object_320_160_npy/'  # raw object array
# train_recon_list_path = '/export/hdd/scratch/dataset/DAC-SDC/INR_cropped_recon_object_DAC_SDC_npy/' # INR reconstructed object array

train_npy_file_path = get_all_folder_paths(train_list_path)
train_ref_npy_file_path = get_all_folder_paths(train_ref_list_path)
train_recon_npy_file_path = get_all_folder_paths(train_recon_list_path)


area_list = []
average_list = []
min_list = []
max_list = []
list_1 = [[] for i in range (object_category_num)]
list_2 = [[] for i in range (object_category_num)]
list_3 = [[] for i in range (object_category_num)]
list_4 = [[] for i in range (object_category_num)]
for i in range (object_category_num):
    area_array = np.load(area_array_path + '/' + "area_info_%d.npy"%(i))
    #temp_list = train_list_new[i].split('/')
    #object_theme = temp_list[-1]
    for area_index in range (area_array.shape[0]):
        if (area_array[area_index] < area_threshold_1):
            list_1[i].append(area_index)
        elif (area_array[area_index] < area_threshold_2):
            list_2[i].append(area_index)
        elif (area_array[area_index] < area_threshold_3):
            list_3[i].append(area_index)
        else:
            list_4[i].append(area_index)


current_object_iter = 0
for i in range (train_object_num):
    if (current_object_iter > 0):
        img_idx_offset = 0
    folder_path = str(train_npy_file_path[object_idx_offset + i])
    ref_folder_path = str(train_ref_npy_file_path[object_idx_offset + i])
    recon_folder_path = str(train_recon_npy_file_path[object_idx_offset + i])
    temp_list = folder_path.split('/')
    ref_teemp_list = ref_folder_path.split('/')
    object_theme = temp_list[-1]
    
    weights_saved_path = INR_encoded_weights_save_path + '/' + object_theme + '/'
    image_array_saved_path = image_object_save_path + '/' + object_theme + '/'
    
    if not os.path.exists(weights_saved_path):
            os.makedirs(weights_saved_path)
            
    if not os.path.exists(image_array_saved_path):
            os.makedirs(image_array_saved_path)
    
    train_files  = get_all_file_paths(folder_path)
    train_ref_files = get_all_file_paths(ref_folder_path)
    recon_files = get_all_file_paths(recon_folder_path)
    #print(len(train_files))
    train_img_num = len(train_files)
    #train_img_num = 1
    for index in range (train_img_num):
    #for index in range (1):
        if img_idx_offset + index in list_1[object_idx_offset + i]:
            layer_size = hidden_dim_class_1
            num_layers = hidden_layer_class_1
            print("this image is type 1")
        elif img_idx_offset + index in list_2[object_idx_offset + i]:
            layer_size = hidden_dim_class_2
            num_layers = hidden_layer_class_2
            print("this image is type 2")
        elif img_idx_offset + index in list_3[object_idx_offset + i]:
            layer_size = hidden_dim_class_3
            num_layers = hidden_layer_class_3
            print("this image is type 3")
        elif img_idx_offset + index in list_4[object_idx_offset + i]:
            layer_size = hidden_dim_class_4
            num_layers = hidden_layer_class_4
            print("this image is type 4")
        print(train_files[img_idx_offset + index])
        img = np.load(train_files[img_idx_offset + index])
        img_ref = np.load(train_ref_files[img_idx_offset + index])
        img_recon = np.load(recon_files[img_idx_offset + index])

        height, width, channel= img.shape
        image_array = np.array(img)
        img_ref_array = np.array(img_ref)
        img_recon_array = np.array(img_recon)
        
        image_array = np.transpose(image_array, (2,1,0))
        
        max_value = np.max(image_array)
        min_value = np.min(image_array)

        Q1 = np.percentile(image_array, 0)  # 10, 90
        Q2 = np.percentile(image_array, 100)

        Q = max(abs(Q1), abs(Q2))

        image_array = image_array/Q
        
        
        func_rep = Siren(
        dim_in=2,
        dim_hidden=layer_size,
        dim_out=3,
        num_layers=num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=w0_initial,
        w0=w0
        ).to(device)
        
        dtype = torch.float32
        img = torch.from_numpy(image_array).to(device, dtype)
        img_ref = torch.from_numpy(img_ref_array).to(device, dtype)
        img_recon = torch.from_numpy(img_recon_array).to(device, dtype)
        
        
        coordinates, features = to_coordinates_and_features(img)
        coordinates, features = coordinates.to(device), features.to(device)
        
        model_size = model_size_in_bits(func_rep) / 8192.
        print(f'Model size: {model_size:.1f}kB')

        time_start_train = time.time()
        trainer = Trainer(func_rep, lr=learning_rate)
        trainer.train(coordinates, features, num_iters, img, Q, img_recon, img_ref/255.0)
        time_stop_train = time.time()
        time_train  = time_stop_train - time_start_train
        train_times.append(time_train)
        total_train_time += train_times[index]
        
        logfile.write(f'{time_train:.18f} ')
        
        logfile.write(f'{trainer.best_vals["psnr"]:.2f}\n')
        
        # if i == train_image_num - 1:
        #     #print(func_rep)
        #     logfile.write(f'\n{func_rep}\n\n')
        #     fp_bpp = bpp(model=func_rep, image=img)
        #     results['fp_bpp'].append(fp_bpp)
        print("the best PNSR is:", trainer.best_vals['psnr'])   
        results['fp_psnr'].append(trainer.best_vals['psnr'])
        avg_psnr += results['fp_psnr'][i]
        torch.save(trainer.current_model, weights_saved_path + f'/best_model_{img_idx_offset + index}.pt')
        #func_rep.load_state_dict(trainer.best_model)


        with torch.no_grad():
            img_recon_features = func_rep(coordinates)
            reconstructured_features = img_recon_features * Q
            reversed_img = reconstructured_features.T.reshape(img.shape)
            reversed_image_tran = reversed_img.permute(2,1,0)
            img_combined = img_recon - reversed_image_tran
            
            
            #img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 1, 0)
            #save_image(torch.clamp(img_recon, 0, 1).to('cpu'), log_dir_image + f'/fp_reconstruction_{i}.png')
            img_array = img_combined.cpu().numpy()
            #print("the shape of the img_array is:",img_array.shape)
            np.save(image_array_saved_path + f'/object_reconstruction_{img_idx_offset + index}.npy', img_array)
            
    current_object_iter += 1
            
            # if not os.path.exists(log_dir_image + dir_appendix):
            #     os.makedirs(log_dir_image + dir_appendix)
            # save_image_PIL(img_recon, log_dir_image + dir_appendix + f'/fp_reconstruction_{i}.png')
    
    

# for i in range (train_image_num):
    
#     #img = cv2.imread(train_files[idx_offset + i])
#     img = np.load(train_files[i])

#     height, width, channel= img.shape
#     print("the height of the image is:", height)
#     print("the width of the image is:", width)
#     image_array = np.array(img)
#     #print("the shape of the image is:", image_array.shape)
#     image_array = np.transpose(image_array, (2,1,0))