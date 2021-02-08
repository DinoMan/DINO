import os
import re
import torch
import random
import torch.nn as nn
from math import ceil
import cv2
import numpy as np


def save_img(img, path):
    save_img = np.squeeze(125 * img.squeeze().cpu().detach().numpy() + 125).astype(np.uint8)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if img.ndim == 2:
        cv2.imwrite(path, save_img)
    else:
        cv2.imwrite(path, cv2.cvtColor(np.rollaxis(save_img, 0, 3), cv2.COLOR_RGB2BGR))


def list_files(folder, exts=None):
    file_list = []
    for root, dirs, files in os.walk(folder, topdown=False):
        if not dirs and files:
            for f in files:
                file_path = os.path.join(root, f)
                if exts is not None and os.path.splitext(file_path)[1]:
                    file_list.append(file_path)

    return file_list


def find_extensions(file_name, allowed_exts=None):
    if allowed_exts is not None:
        if not isinstance(allowed_exts, list):
            allowed_exts = [allowed_exts]

        ext_list = []
        for ext in allowed_exts:
            if os.path.isfile(file_name + ext):
                ext_list.append(ext)

        return ext_list

    path = os.path.dirname(file_name)
    name = os.path.basename(file_name)

    files = os.listdir(path)
    rx = r'({0}\.\w+)'.format(name)

    matched_files = re.findall(rx, " ".join(files))
    return [os.path.splitext(m)[1] for m in matched_files]


def list_matching_files(directories, ext=None):
    matching_files = []
    matching_dirs = []
    matching_exts = []
    for root, dirs, files in os.walk(directories[0], topdown=False):
        for name in files:
            file_name = os.path.splitext(name)[0]
            extensions = [os.path.splitext(name)[1]]

            sub_path = root.replace(directories[0], "")
            if sub_path != '' and sub_path[0] == '/':
                sub_path = sub_path[1:]

            # Now search for matching files in the other directories
            all_matches_found = True
            for i in range(1, len(directories)):

                # Find all the possible extensions of the file in the directory
                if ext is None:
                    possible_extensions = find_extensions(os.path.join(directories[i], sub_path, file_name))
                else:
                    possible_extensions = find_extensions(os.path.join(directories[i], sub_path, file_name), ext[i])

                if not possible_extensions:  # If you can't find any files
                    all_matches_found = False  # Report that not all matches were found and break
                    break

                # If we have found files with that name
                match_found = False
                for possible_extension in possible_extensions:  # Check all possible extensions to find only valid ones
                    if os.path.isfile(os.path.join(directories[i], sub_path, file_name + possible_extension)):
                        match_found = True  # If we foind a suitable one then report and break
                        extensions.append(possible_extension)
                        break

                if not match_found:  # If no match was found report that not all matches were found
                    all_matches_found = False
                    break

            if not all_matches_found:
                continue

            matching_dirs.append(sub_path)
            matching_files.append(file_name)
            matching_exts.append(extensions)

    return {"files": matching_files, "dirs": matching_dirs, "exts": matching_exts}


def same_padding(kernel_size, stride=1, in_size=0):
    out_size = ceil(float(in_size) / float(stride))
    return int((out_size - 1) * stride + kernel_size - in_size)


def initialization(weights, type='xavier', init=None):
    if type == 'normal':
        if init is None:
            torch.nn.init.normal_(weights)
        else:
            torch.nn.init.normal_(weights, mean=init[0], std=init[1])
    elif type == 'xavier':
        if init is None:
            torch.nn.init.xavier_normal_(weights)
        else:
            torch.nn.init.xavier_normal_(weights, gain=init)
    elif type == 'kaiming':
        torch.nn.init.kaiming_normal_(weights)
    elif type == 'orthogonal':
        if init is None:
            torch.nn.init.orthogonal_(weights)
        else:
            torch.nn.init.orthogonal_(weights, gain=init)
    else:
        raise NotImplementedError('Unknown initialization method')


def initialize_weights(net, type='xavier', init=None, init_bias=False, batchnorm_shift=None):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            initialization(m.weight, type=type, init=init)
            if init_bias and hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1 and batchnorm_shift is not None:
            torch.nn.init.normal_(m.weight, 1.0, batchnorm_shift)
            torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
            for layer_params in m._all_weights:
                for param in layer_params:
                    if 'weight' in param:
                        initialization(m._parameters[param])

    net.apply(init_func)


class ResizeConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2, bias=True, mode='nearest', spectral_norm=False):
        super(ResizeConv2D, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)
        if kernel_size % 2 == 1:
            padding = kernel_size // 2
        else:
            padding = (kernel_size // 2, (kernel_size // 2) - 1, kernel_size // 2, (kernel_size // 2) - 1)
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias)
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x, output_size=None):
        x = self.up(x)
        x = self.pad(x)
        x = self.conv(x)

        if output_size is not None:
            return x.view(output_size)

        return x


class UnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, in_size, kernel_size, stride=1, norm=None, spectral_norm=False, bias=False,
                 activation=None, activation_params=[], resize_convs=False, dropout=0):
        super(UnetBlock2D, self).__init__()

        self.dropout1 = None
        self.dropout2 = None

        if dropout > 0:
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)

        if activation is None:
            activation = nn.ReLU

        # This ensures that we have same padding no matter if we have even or odd kernels
        padding = same_padding(kernel_size, stride)
        if spectral_norm:
            if resize_convs:
                self.dcl1 = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=bias))
                self.dcl2 = ResizeConv2D(in_channels, out_channels, kernel_size, scale_factor=stride, bias=bias, spectral_norm=True)
            else:
                self.dcl1 = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=bias))
                self.dcl2 = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                                      stride=stride, padding=padding // 2, bias=bias))
        else:
            if resize_convs:
                self.dcl1 = nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=bias)
                self.dcl2 = ResizeConv2D(in_channels, out_channels, kernel_size, scale_factor=stride, bias=bias)
            else:
                self.dcl1 = nn.ConvTranspose2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=bias)
                self.dcl2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding // 2, bias=bias)

        if norm is not None:
            self.activation1 = nn.Sequential(norm(in_channels), activation(*activation_params))
            self.activation2 = nn.Sequential(norm(out_channels), activation(*activation_params))
        else:
            self.activation1 = activation(*activation_params)
            self.activation2 = activation(*activation_params)

        self.required_channels = out_channels
        self.out_size_required = tuple(x * stride for x in in_size)

    def forward(self, x, s):
        s = s.view(x.size())

        x = torch.cat([x, s], 1)

        x = self.dcl1(x)
        x = self.activation1(x)
        if self.dropout1 is not None:
            x = self.dropout1(x)

        x = self.dcl2(x, output_size=[-1, self.required_channels, self.out_size_required[0], self.out_size_required[1]])
        x = self.activation2(x)
        if self.dropout2 is not None:
            x = self.dropout2(x)
        return x


class RandomCrop(object):
    def __init__(self, proportion=0.9):
        if not isinstance(proportion, tuple):
            self.proportion = (proportion, proportion)

    def __call__(self, source, proportion=None):
        if proportion is None:
            proportion = self.proportion

        try:  # If img is iterable
            img_iterator = iter(source)
        except TypeError:
            img_iterator = iter([source])

        tl_ratio_x = random.uniform(0, 1)
        tl_ratio_y = random.uniform(0, 1)

        target = []
        for img in img_iterator:
            w, h = img.size
            new_w = proportion[0] * w
            new_h = proportion[1] * h

            x_tl = int(tl_ratio_x * (w - new_w))
            y_tl = int(tl_ratio_y * (h - new_h))
            target.append(img.crop((x_tl, y_tl, x_tl + new_w, y_tl + new_h)))

        if len(target) == 1:
            return target[0]
        else:
            return target


def filify(string):
    filename = string.replace(" ", "_")
    filename = filename.replace(":", "-")
    filename = filename.replace("-_", "-")
    return filename


def dict2args(dictionary):
    arg_list = []
    for k in dictionary.keys():
        if type(dictionary[k]) == type(True):
            if dictionary[k] == True:
                arg_list += ["--" + k]
            continue
        elif dictionary[k] is None:
            continue
        elif isinstance(dictionary[k], list):
            arg_list += ["--" + k]
            for subvalue in dictionary[k]:
                arg_list += [subvalue]
            continue

        arg_list += ["--" + k, str(dictionary[k])]
    return arg_list
