import numpy as np
import torch.nn as nn
import torch
import utils
import itertools
import copy


class Encoder(nn.Module):
    def __init__(self, img_size, code_size, kernel_size=4, num_input_channels=3, num_feature_maps=64, max_feature_maps=512, norm=None,
                 use_bias=False):
        super(Encoder, self).__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)

        self.final_size = (4, 4)

        self.code_size = code_size
        self.num_feature_maps = num_feature_maps
        self.max_feature_maps = max_feature_maps
        self.cl = nn.ModuleList()
        self.num_layers = int(np.log2(max(self.img_size))) - 2

        stride = 2
        # This replicates the same padding functionality from Tensorflow
        padding = utils.same_padding(kernel_size, stride)
        if norm is None:
            norm = nn.Identity

        # The first layer is not automatically calculated
        self.cl.append(nn.Sequential(
            nn.Conv2d(num_input_channels, self.num_feature_maps, kernel_size, stride=stride, padding=padding // 2, bias=use_bias),
            norm(self.num_feature_maps),
            nn.LeakyReLU(0.05, inplace=True)))

        self.channels = [self.num_feature_maps]
        for i in range(self.num_layers - 1):
            out_channels = min(max_feature_maps, 2 * self.channels[-1])
            self.cl.append(nn.Sequential(
                nn.Conv2d(self.channels[-1], out_channels, kernel_size, stride=stride, padding=padding // 2, bias=use_bias),
                norm(out_channels),
                nn.LeakyReLU(0.05, inplace=True)))
            self.channels.append(out_channels)

        self.cl.append(
            nn.Sequential(nn.Conv2d(self.channels[-1], code_size, self.final_size, stride=1, padding=0, bias=use_bias), nn.Tanh()))

        # Initialise the weights for the network with Gaussians with 0 mean and 0.02 std. dev
        utils.initialize_weights(self, type="normal", init=[0.0, 0.02])

    def forward(self, x, retain_intermediate=False):
        if retain_intermediate:
            h = [x]
            for conv_layer in self.cl:
                h.append(conv_layer(h[-1]))
            return h[-1].view(-1, self.code_size), h[1:-1]
        else:
            for conv_layer in self.cl:
                x = conv_layer(x)

            return x.view(-1, self.code_size)


class Decoder(nn.Module):
    def __init__(self, img_size, latent_size, noise_size=0, kernel_size=3, num_img_channels=3, num_gen_channels=512, min_feat_maps=64,
                 skip_channels=[], mirror_skips=True, norm=None, use_bias=False, dual_head_layer=None):
        super(Decoder, self).__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)

        self.init_size = (4, 4)

        self.latent_size = latent_size
        self.noise_size = noise_size
        self.min_feat_maps = min_feat_maps
        self.num_layers = int(np.log2(max(self.img_size))) - 1
        self.num_img_channels = num_img_channels
        self.num_gen_channels = num_gen_channels
        self.dcl = nn.ModuleList()

        # If we require a dual head network then create the secondary stream
        self.dual_head_layer = dual_head_layer
        if self.dual_head_layer is not None:
            self.dcl2 = nn.ModuleList()

        self.total_latent_size = self.latent_size + self.noise_size

        if norm is None:
            norm = nn.Identity

        stride = 2
        self.dcl.append(nn.Sequential(
            nn.ConvTranspose2d(self.total_latent_size, num_gen_channels, self.init_size, bias=use_bias),
            norm(num_gen_channels),
            nn.ReLU(inplace=True)))

        # If splitting happens right at the beginning then copy the first layer to the second stream
        if self.dual_head_layer is not None and self.dual_head_layer == 0:
            self.dcl2.append(copy.deepcopy(self.dcl[-1]))

        self.channels = [num_gen_channels]
        in_size = self.init_size

        for i in range(1, self.num_layers - 1):
            if skip_channels and mirror_skips:
                num_out_channels = skip_channels[i]
            else:
                num_out_channels = max(self.channels[-1] // 2, self.min_feat_maps)

            if not skip_channels:
                self.dcl.append(nn.Sequential(
                    utils.ResizeConv2D(self.channels[-1], num_out_channels, kernel_size, scale_factor=stride, bias=use_bias),
                    norm(num_out_channels),
                    nn.ReLU(inplace=True)))
            else:
                self.dcl.append(utils.UnetBlock2D(self.channels[-1], num_out_channels, skip_channels[i - 1], in_size,
                                                kernel_size, stride=stride, norm=norm, activation=nn.ReLU, activation_params=[True],
                                                resize_convs=True, bias=use_bias))

            if self.dual_head_layer is not None and i >= self.dual_head_layer:
                self.dcl2.append(copy.deepcopy(self.dcl[-1]))  # Copy the layer to the dual head

            # Number of inputs will be the same as number of previous layer outputs
            self.channels.append(num_out_channels)
            in_size = tuple(2 * x for x in in_size)

        self.dcl.append(utils.ResizeConv2D(self.channels[-1], self.num_img_channels, kernel_size, scale_factor=stride, bias=use_bias))

        # Copy the final layer of the primary stream to the secondary stream
        if self.dual_head_layer is not None and self.dual_head_layer < self.num_layers:
            self.dcl2.append(copy.deepcopy(self.dcl[-1]))

        self.final_activation = nn.Tanh()

        # Initialise the weights for the network with Gaussians with 0 mean and 0.02 std. dev
        utils.initialize_weights(self, type="normal", init=[0.0, 0.02])

    def primary_stream_params(self):
        params = [self.dcl.parameters()]
        return itertools.chain(*params)

    def secondary_stream_params(self):
        if self.dual_head_layer is None:
            return self.primary_stream_params()
        else:
            return itertools.chain(self.dcl[:self.dual_head_layer].parameters(), self.dual_head_params())

    def primary_head_params(self):
        params = []
        if self.dual_head_layer is not None:
            params += [self.dcl[self.dual_head_layer:].parameters()]
        else:
            return self.parameters()

        return itertools.chain(*params)

    def dual_head_params(self):
        params = []
        if self.dual_head_layer is not None:
            params += [self.dcl2.parameters()]
        else:
            return self.parameters()

        return itertools.chain(*params)

    def forward(self, x, skip=[], n=None, dual=False):
        if n is not None:
            x = torch.cat([x, n], 1)

        x = x.view(-1, self.total_latent_size, 1, 1)
        x = self.dcl[0](x)

        for i in range(1, self.num_layers - 1):
            if dual and self.dual_head_layer is not None and i >= self.dual_head_layer:
                deconv = self.dcl2[i - self.dual_head_layer]
            else:
                deconv = self.dcl[i]

            if not skip:
                x = deconv(x)
            else:
                layer_size = list(x.size())
                layer_size[1] = -1
                x = deconv(x, skip[i - 1].view(layer_size))

        if dual and self.dual_head_layer is not None and self.dual_head_layer < self.num_layers:
            deconv = self.dcl2[-1]
        else:
            deconv = self.dcl[-1]

        x = deconv(x, output_size=[-1, self.num_img_channels, self.img_size[0], self.img_size[1]])
        return self.final_activation(x)


class EncoderDecoder(nn.Module):
    def __init__(self, img_size, latent_size, noise_size=0, num_input_channels=3, norm="batch", use_skips=True, use_bias=False, dual_head_layer=None):
        super(EncoderDecoder, self).__init__()

        if norm == "batch":
            norm = nn.BatchNorm2d
        elif norm == "instance":
            norm = nn.InstanceNorm2d
        else:
            norm = None

        self.noise_size = noise_size
        self.use_skips = use_skips
        self.enc = Encoder(img_size, latent_size, num_input_channels=num_input_channels, norm=norm, use_bias=use_bias)

        if self.use_skips:
            skip_channels = list(self.enc.channels)
            skip_channels.reverse()
        else:
            skip_channels = []

        self.dec = Decoder(img_size, latent_size, noise_size=noise_size, skip_channels=skip_channels, norm=norm, use_bias=use_bias,
                           dual_head_layer=dual_head_layer)

    def forward(self, x, n=None, dual=False):
        if self.use_skips:
            x, skips = self.enc(x, retain_intermediate=True)
            return self.dec(x, n=n, skip=list(reversed(skips)), dual=dual)
        else:
            x = self.enc(x)
            return self.dec(x, n=n, dual=dual)
