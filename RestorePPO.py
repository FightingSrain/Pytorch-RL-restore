import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PPO(object):
    def __init__(self, memory, acnet, clip_param_start, clip_param_now,
                 num_epi, clip_linear_jud, max_grad_norm,
                 ppo_update_time, buffer_capacity, learn_start,
                 batch_size, learning_rate, gamma):
        self.acnet = acnet
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.clip_param_now = clip_param_now
        self.memory = memory
        self.counter = 0
        self.training_step = 0
        self.optimizers = optim.Adam(self.acnet.parameters(), self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers, 300, eta_min=0.0001, last_epoch=-1)

    def select_action(self, state, pre_action, batch_size=1, ht=None, ct=None, test=False):
        with torch.no_grad():
            action_prob,_, hc, cc = self.acnet(state, pre_action, batch_size, 1, ht=ht, ct=ct, test=False)
        c = Categorical(action_prob)
        action = c.sample()

        return action.item(), action_prob[:, action.item()].item(), action_prob, hc, cc

    def update(self):
        loss = 0
        actorloss = 0
        criticloss = 0
        state, action, reward, old_action_log_prob , done = self.memory.getEpiBatch()

        for o in range(1):
            for m in range(3):
                # 获取第m步的状态，动作，奖励
                s_t_cur = state[m]
                action_cur = action[m]
                reward_cur = reward[m]
                policy_old = old_action_log_prob[m]
                done_cur = done[m]
                num = len(reward_cur) # 奖励的个数。
                if num < 1:
                    continue
                rnn_length = m + 1
                rnn_batch = num // rnn_length
                # derive action at previous step 派生上一步的操作
                action_in = np.zeros([num, 12])
                for k in range(rnn_batch):
                    for p in range(rnn_length):
                        if p > 0:
                            idx = k * rnn_length + p
                            action_in[idx, action_cur[idx - 1]] = 1.
                R = 0
                Gt = []
                for i in reversed(range(int(len(reward_cur)))):
                    R = reward_cur[i] + self.gamma * R
                    Gt.insert(0, R)
                    if done_cur[i] == True and i != (len(reward_cur)-1):
                        R = 0
                Gt = torch.FloatTensor(Gt).cuda()
                s_t_cur = torch.FloatTensor(s_t_cur.transpose(0, 3, 1, 2)).cuda()
                action_in = torch.FloatTensor(action_in).cuda()
                action_cur = torch.LongTensor(action_cur).cuda()
                policy_old = torch.FloatTensor(policy_old).cuda()

                Gt_index = Gt.view(-1, 1)
                action_prob, V,_,_ = self.acnet(s_t_cur, action_in, s_t_cur.size(0), length=rnn_length, off=True)
                delta = Gt_index - V
                advantage = delta.detach()
                if advantage.size(0) > 1:
                    advantage = (advantage - advantage.mean(0).item())/(advantage.std(0).item() + 1e-10)
                action_cur = action_cur.view(s_t_cur.size(0), 1)
                action_prob = action_prob.gather(1, action_cur)  # new policy

                ratio = (action_prob / policy_old)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param_now, 1 + self.clip_param_now) * advantage

                loss_entropy = torch.mean(torch.log(action_prob) * action_prob)
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(Gt_index.detach(), V)
                loss = action_loss + 0.5 * value_loss + 0.01 * loss_entropy
                actorloss = action_loss
                criticloss = value_loss
                self.optimizers.zero_grad()
                loss.backward()
                self.optimizers.step()
                self.training_step += 1
        return loss, actorloss, criticloss