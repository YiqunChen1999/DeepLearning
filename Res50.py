
def set_res50_params():
    print(INFO, 'setting parameters of Res50...')
    
    conv_2_1_params = {
    'input_channels': 256, 
    'output_channels': 64
    }
    conv_2_2_params = {
    'input_channels': 64, 
    'output_channels': 64
    }
    conv_2_3_params = {
    'input_channels': 64, 
    'output_channels': 256
    }
    conv2_x_params = {
    'conv1': conv_2_1_params, 
    'conv2': conv_2_2_params, 
    'conv3': conv_2_3_params
    }

    conv_3_1_params = {
    'input_channels': 512, 
    'output_channels': 128
    }
    conv_3_2_params = {
    'input_channels': 128, 
    'output_channels': 128
    }
    conv_3_3_params = {
    'input_channels': 128, 
    'output_channels': 512
    }
    conv3_x_params = {
    'conv1': conv_3_1_params, 
    'conv2': conv_3_2_params, 
    'conv3': conv_3_3_params
    }

    conv_4_1_params = {
    'input_channels': 1024, 
    'output_channels': 256
    }
    conv_4_2_params = {
    'input_channels': 256, 
    'output_channels': 256
    }
    conv_4_3_params = {
    'input_channels': 256, 
    'output_channels': 1024
    }
    conv4_x_params = {
    'conv1': conv_4_1_params, 
    'conv2': conv_4_2_params, 
    'conv3': conv_4_3_params
    }

    conv_5_1_params = {
    'input_channels': 2048, 
    'output_channels': 512
    }
    conv_5_2_params = {
    'input_channels': 512, 
    'output_channels': 512
    }
    conv_5_3_params = {
    'input_channels': 512, 
    'output_channels': 2048
    }
    conv5_x_params = {
    'conv1': conv_5_1_params, 
    'conv2': conv_5_2_params, 
    'conv3': conv_5_3_params
    }

    res50_params = {
    'conv2_x': conv2_x_params, 
    'conv3_x': conv3_x_params, 
    'conv4_x': conv4_x_params, 
    'conv5_x': conv5_x_params
    }

    return res50_params


class ResBlock(nn.Module):
    def __init__(self, params, size_down=False):
        super(ResBlock, self).__init__()
        self.params = params
        self.size_down = size_down
        if self.size_down:
            conv1_stride = 2
        else:
            conv1_stride = 1
        self.conv1 = nn.Conv2d(
            in_channels=self.params['conv1']['input_channels'], 
            out_channels=self.params['conv1']['output_channels'], 
            kernel_size=1, 
            stride=conv1_stride, 
            padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=self.params['conv2']['input_channels'], 
            out_channels=self.params['conv2']['output_channels'], 
            kernel_size=3, 
            padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=self.params['conv3']['input_channels'], 
            out_channels=self.params['conv3']['output_channels'], 
            kernel_size=1, 
            padding=0)
        self.conv_proj = nn.Conv2d(
            in_channels=self.params['conv1']['input_channels'], 
            out_channels=self.params['conv3']['output_channels'], 
            kernel_size=1, 
            stride=conv1_stride,
            padding=0)
        self.norm1 = nn.BatchNorm2d(
            num_features=self.params['conv1']['output_channels'])
        self.norm2 = nn.BatchNorm2d(
            num_features=self.params['conv2']['output_channels'])
        self.norm3 = nn.BatchNorm2d(
            num_features=self.params['conv3']['output_channels'])
        self.norm_proj = nn.BatchNorm2d(
            num_features=self.params['conv3']['output_channels'])
        self.norm = nn.BatchNorm2d(
            num_features=self.params['conv3']['output_channels'])

    def forward(self, x):
        #print(INFO, x.shape)
        identity_x = self.conv_proj(x)
        #print(INFO, x.shape)
        identity_x = self.norm_proj(identity_x)
        #print(INFO, x.shape)
        x = self.conv1(x)
        #print(INFO, x.shape)
        x = self.norm1(x)
        #print(INFO, x.shape)
        x = F.relu(x)
        #print(INFO, x.shape)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = x + identity_x
        x = self.norm(x)
        x = F.relu(x)
        return x


class Res50(nn.Module):
    def __init__(self, res50_params):
        super(Res50, self).__init__()
        print(INFO, 'constructing the Res50...')
        self.res50_params = res50_params
        
        # construct the conv1
        self.conv1_x = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=1)
        self.norm1 = nn.BatchNorm2d(
            num_features=64)

        # construct the conv2_x
        self.conv2_x_params = self.res50_params['conv2_x']
        
        conv1_params = {
        'input_channels': 64, 
        'output_channels': 64
        }
        conv2_params = {
        'input_channels': 64, 
        'output_channels': 64
        }
        conv3_params = {
        'input_channels': 64, 
        'output_channels': 256
        }
        first_block_params = {
        'conv1': conv1_params, 
        'conv2': conv2_params, 
        'conv3': conv3_params
        }

        self.conv2_x_1 = ResBlock(
            params=first_block_params,
            size_down=False)
        self.conv2_x_2 = ResBlock(
            params=self.conv2_x_params, 
            size_down=False)
        self.conv2_x_3 = ResBlock(
            params=self.conv2_x_params, 
            size_down=True)

        # construct the conv3_x
        self.conv3_x_params = self.res50_params['conv3_x']
        
        conv1_params = {
        'input_channels': 256, 
        'output_channels': 128
        }
        conv2_params = {
        'input_channels': 128, 
        'output_channels': 128
        }
        conv3_params = {
        'input_channels': 128, 
        'output_channels': 512
        }
        first_block_params = {
        'conv1': conv1_params, 
        'conv2': conv2_params, 
        'conv3': conv3_params
        }

        self.conv3_x_1 = ResBlock(
            params=first_block_params, 
            size_down=False)
        self.conv3_x_2 = ResBlock(
            params=self.conv3_x_params, 
            size_down=False)
        self.conv3_x_3 = ResBlock(
            params=self.conv3_x_params, 
            size_down=False)
        self.conv3_x_4 = ResBlock(
            params=self.conv3_x_params,
            size_down=True)

        # construct the conv4_x
        self.conv4_x_params = self.res50_params['conv4_x']

        conv1_params = {
        'input_channels': 512, 
        'output_channels': 256
        }
        conv2_params = {
        'input_channels': 256, 
        'output_channels': 256
        }
        conv3_params = {
        'input_channels': 256, 
        'output_channels': 1024
        }
        first_block_params = {
        'conv1': conv1_params, 
        'conv2': conv2_params, 
        'conv3': conv3_params
        }

        self.conv4_x_1 = ResBlock(
            params=first_block_params, 
            size_down=False)
        self.conv4_x_2 = ResBlock(
            params=self.conv4_x_params, 
            size_down=False)
        self.conv4_x_3 = ResBlock(
            params=self.conv4_x_params, 
            size_down=False)
        self.conv4_x_4 = ResBlock(
            params=self.conv4_x_params, 
            size_down=False)
        self.conv4_x_5 = ResBlock(
            params=self.conv4_x_params, 
            size_down=False)
        self.conv4_x_6 = ResBlock(
            params=self.conv4_x_params, 
            size_down=True)

        # construct the conv5_x.
        self.conv5_x_params = self.res50_params['conv5_x']

        conv1_params = {
        'input_channels': 1024, 
        'output_channels': 512
        }
        conv2_params = {
        'input_channels': 512, 
        'output_channels': 512
        }
        conv3_params = {
        'input_channels': 512, 
        'output_channels': 2048
        }
        first_block_params = {
        'conv1': conv1_params, 
        'conv2': conv2_params, 
        'conv3': conv3_params
        }

        self.conv5_x_1 = ResBlock(
            params=first_block_params,
            size_down=False)
        self.conv5_x_2 = ResBlock(
            params=self.conv5_x_params, 
            size_down=False)
        self.conv5_x_3 = ResBlock(
            params=self.conv5_x_params, 
            size_down=True)

        # construct the last stage.
        self.global_ave_pooling = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            in_features=2048, 
            out_features=10)

        print(INFO, 'done!')

    def forward(self, x):
        #print(x.shape)
        x = self.conv1_x(x)
        #print(x.shape)
        x = self.norm1(x)
        #print(x.shape)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv2_x_1(x)
        #print(x.shape)
        x = self.conv2_x_2(x)
        #print(x.shape)
        x = self.conv2_x_3(x)
        x = self.conv3_x_1(x)
        x = self.conv3_x_2(x)
        x = self.conv3_x_3(x)
        x = self.conv3_x_4(x)
        x = self.conv4_x_1(x)
        x = self.conv4_x_2(x)
        x = self.conv4_x_3(x)
        x = self.conv4_x_4(x)
        x = self.conv4_x_5(x)
        x = self.conv4_x_6(x)
        x = self.conv5_x_1(x)
        x = self.conv5_x_2(x)
        x = self.conv5_x_3(x)
        x = self.global_ave_pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.relu(x)
        #x = F.softmax(x)
        return x

# class Res50(nn.Module):
#     def __init__(self):
#         super(Res50, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=3, 
#             out_channels=64, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)


#         self.conv211 = nn.Conv2d(
#             in_channels=64, 
#             out_channels=64, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv212 = nn.Conv2d(
#             in_channels=64, 
#             out_channels=64, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv213 = nn.Conv2d(
#             in_channels=64, 
#             out_channels=256, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
        
#         self.conv221 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=64, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv222 = nn.Conv2d(
#             in_channels=64, 
#             out_channels=64, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv223 = nn.Conv2d(
#             in_channels=64, 
#             out_channels=256, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)

#         self.conv231 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=64, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv232 = nn.Conv2d(
#             in_channels=64, 
#             out_channels=64, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv233 = nn.Conv2d(
#             in_channels=64, 
#             out_channels=256, 
#             kernel_size=1, 
#             stride=2, 
#             padding=0)

#         self.conv311 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=64, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv312 = nn.Conv2d(
#             in_channels=128, 
#             out_channels=128, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv313 = nn.Conv2d(
#             in_channels=128, 
#             out_channels=512, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
        
#         self.conv321 = nn.Conv2d(
#             in_channels=512, 
#             out_channels=128, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv322 = nn.Conv2d(
#             in_channels=128, 
#             out_channels=128, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv323 = nn.Conv2d(
#             in_channels=128, 
#             out_channels=512, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)

#         self.conv331 = nn.Conv2d(
#             in_channels=512, 
#             out_channels=128, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv332 = nn.Conv2d(
#             in_channels=128, 
#             out_channels=128, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv333 = nn.Conv2d(
#             in_channels=128, 
#             out_channels=512, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)

#         self.conv341 = nn.Conv2d(
#             in_channels=512, 
#             out_channels=128, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv342 = nn.Conv2d(
#             in_channels=128, 
#             out_channels=128, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv343 = nn.Conv2d(
#             in_channels=128, 
#             out_channels=512, 
#             kernel_size=1, 
#             stride=2, 
#             padding=0)


#         self.conv411 = nn.Conv2d(
#             in_channels=512, 
#             out_channels=256, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv412 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=256, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv413 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=1024, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
        
#         self.conv421 = nn.Conv2d(
#             in_channels=1024, 
#             out_channels=256, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv422 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=256, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv423 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=1024, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)

#         self.conv431 = nn.Conv2d(
#             in_channels=1024, 
#             out_channels=256, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv432 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=256, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv433 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=1024, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)

#         self.conv441 = nn.Conv2d(
#             in_channels=1024, 
#             out_channels=256, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv442 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=256, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv443 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=1024, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)

#         self.conv451 = nn.Conv2d(
#             in_channels=1024, 
#             out_channels=256, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv452 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=256, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv453 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=1024, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)

#         self.conv461 = nn.Conv2d(
#             in_channels=1024, 
#             out_channels=256, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv462 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=256, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv463 = nn.Conv2d(
#             in_channels=256, 
#             out_channels=1024, 
#             kernel_size=1, 
#             stride=2, 
#             padding=0)

#         self.conv511 = nn.Conv2d(
#             in_channels=1024, 
#             out_channels=512, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv512 = nn.Conv2d(
#             in_channels=512, 
#             out_channels=512, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv513 = nn.Conv2d(
#             in_channels=512, 
#             out_channels=2048, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
        
#         self.conv521 = nn.Conv2d(
#             in_channels=2048, 
#             out_channels=512, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv522 = nn.Conv2d(
#             in_channels=512, 
#             out_channels=512, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv523 = nn.Conv2d(
#             in_channels=512, 
#             out_channels=2048, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)

#         self.conv531 = nn.Conv2d(
#             in_channels=2028, 
#             out_channels=512, 
#             kernel_size=1, 
#             stride=1, 
#             padding=0)
#         self.conv532 = nn.Conv2d(
#             in_channels=512, 
#             out_channels=512, 
#             kernel_size=3, 
#             stride=1, 
#             padding=1)
#         self.conv533 = nn.Conv2d(
#             in_channels=512, 
#             out_channels=2048, 
#             kernel_size=1, 
#             stride=2, 
#             padding=0)

#         self.global_ave_pooling = nn.AvgPool2d(kernel_size=2)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(
#             in_features=2048, 
#             out_features=10)

