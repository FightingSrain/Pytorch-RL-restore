import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ACnet(nn.Module):
    def __init__(self):
        super(ACnet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=2, padding=(4, 4), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=5, stride=2, padding=(2, 2), bias=True),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=2, padding=(2, 2), bias=True),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=2, padding=(2, 2), bias=True),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.actor = nn.Softmax(dim=1)
        self.flat_actor = nn.Linear(384,32)
        self.actor_re = nn.LeakyReLU()
        self.critic_re = nn.LeakyReLU()
        self.flat_critic01 = nn.Linear(50, 50)
        self.flat_critic0 = nn.Linear(50, 32)
        self.flat_critic1 = nn.Linear(32,1)
        self.flat_lstm_a0 = nn.Linear(50, 50)
        self.flat_lstm_a = nn.Linear(50, 32)
        self.flat_lstm_a_ = nn.Linear(32, 13)
        self.flat_re = nn.LeakyReLU()
        self.lstm = nn.LSTM(44, 50, 1)

        self.flat_critic0.apply(self.weight_init)
        self.flat_critic1.apply(self.weight_init)
        self.flat_actor.apply(self.weight_init)
        self.flat_lstm_a0.apply(self.weight_init)
        self.flat_lstm_a.apply(self.weight_init)
        self.flat_lstm_a_.apply(self.weight_init)
        self.lstm.apply(self.weight_init)

    def forward(self, x, pre_action, batch_size=1, length=1, ht=None, ct=None,test=False,off=False):
        conv_out = self.conv(x)
        conv_out = conv_out.view(batch_size, -1)
        conv_out = self.flat_actor(conv_out)
        conv_out = F.leaky_relu(conv_out)
        ###########
        if off == False:
            pre_act_one_hot = self.get_one_hot(pre_action, batch_size)
            l_in = torch.cat([pre_act_one_hot.cuda(), conv_out], dim=1).unsqueeze(0)
        else:
            l_in = torch.cat([pre_action.cuda(), conv_out], dim=1).unsqueeze(0)
        l_in = l_in.view(int(length), batch_size // int(length), 44)
        if test == False:
            h0 = torch.zeros(1, batch_size // int(length), 50)
            c0 = torch.zeros(1, batch_size // int(length), 50)
        else:
            h0 = ht
            c0 = ct
        output, (hn, cn) = self.lstm(l_in, (h0.cuda(), c0.cuda()))
        output = output.view(-1, 50)
        #################################
        outputa = self.flat_lstm_a0(output)
        outputa = F.layer_norm(outputa, (output.size(0), 50))
        outputa = F.leaky_relu(outputa)
        


        outputa = self.flat_lstm_a(outputa)
        outputa = F.layer_norm(outputa,(outputa.size(0),32))
        outputa = F.leaky_relu(outputa)
        
        outputa = self.flat_lstm_a_(outputa)
        policy = F.softmax(outputa, dim=1)
        ###########
        value = self.flat_critic01(output)
        value = F.layer_norm(value, (output.size(0), 50))
        value = F.leaky_relu(value)
        

        value = self.flat_critic0(value)
        value = F.layer_norm(value, (output.size(0), 32))
        value = F.leaky_relu(value)
        
        value = self.flat_critic1(value)
        return policy, value, hn, cn

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def get_one_hot(self, pre_action, batch_size):
        one_hots = torch.zeros(batch_size, 12)

        for i in range(batch_size):
            if pre_action[i].item() == -1:
                continue
            else:
                one_hots[i, pre_action[i].item()] = 1
        return one_hots

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())
