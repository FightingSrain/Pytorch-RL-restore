
import numpy as np
import matplotlib.pyplot as plt
import torch
from environments import Env
from replay_memorys import ReplayMemory
from ACnet import ACnet
from RestorePPO import PPO
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

clip_param_start = 1.5 # 1.5 # 0.2
clip_param_now = 1.5
num_epi = 1000000
clip_linear_jud = False
max_grad_norm = 0.5
ppo_update_time = 10
buffer_capacity = 100000
learn_start = 32
batch_size = 32
learning_rate = 0.0001
gamma = 0.98
memory_size = 100000

train_dir = 'dataset/train/'
vali_dir = 'dataset/valid/'
test_dir = 'dataset/test/'
dataset = 'moderate'
def main():
    env = Env(train_dir, vali_dir, test_dir, dataset)
    acnet = ACnet().to(device)
    memory = ReplayMemory()
    agent = PPO(memory, acnet, clip_param_start, clip_param_now,
                num_epi, clip_linear_jud, max_grad_norm,
                ppo_update_time, buffer_capacity, learn_start,
                batch_size, learning_rate, gamma)

    sum_reward = []
    total_loss = []
    actor_loss = []
    critic_loss = []
    weight_reward = -1
    max_reward = 0
    img, _, _, terminal = env.new_image()
    for i_epoch in tqdm(range(0, num_epi), ncols=70, initial=0):

        agent.scheduler.step()
        agent.learning_rate = agent.scheduler.get_lr()

        pre_action = -1
        total_reward = 0
        h0x = torch.zeros(1, 1, 50)
        c0x = torch.zeros(1, 1, 50)
        while True:
            img = np.clip(img, a_max=0.99, a_min=0.)

            action, action_prob, policy, hc, cc = agent.select_action(torch.FloatTensor(img.transpose(0, 3, 1, 2)).to(device),
                                                      torch.LongTensor([pre_action]).unsqueeze(1).cuda(), ht=h0x, ct=c0x, test=True)
            img, reward, done = env.step(action)
            total_reward += reward
            memory.add(img, reward, action, action_prob, done)
            pre_action = action
            h0x = hc
            c0x = cc
            if done:
                img, _, _, terminal = env.new_image()
                if weight_reward==-1:
                    weight_reward = total_reward
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * total_reward
                if i_epoch >= learn_start and i_epoch % 1 == 0: # <---------- 增加探索
                    te_loss, a_loss, c_loss= agent.update()      #   <-----------
                    total_loss.append(te_loss)
                    actor_loss.append(a_loss)
                    critic_loss.append(c_loss)
                # agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                # print('episode: {}  reward: {}  weight_reward: {:.2f}'.format(i_epoch + 1, total_reward, weight_reward))
                start_policy = np.array([0.07692307] * 13)
                memory.add(img, 0., -1, 0.07692307, terminal)
                break
        # if i_epoch % 500 == 0 and i_epoch > agent.learn_start:
        #     print("lR:", agent.learning_rate)
        #     torch.save(agent.acnet, 'ppo_model.pkl')
        if i_epoch % 100 == 0 and i_epoch > learn_start:
            reward_test_vec = []
            psnr_test_vec = []
            reward_str = ''
            psnr_str = ''
            reward_sum = 0.
            pre_action_test = torch.full((160, 1), -1).long()
            h0 = torch.zeros(1, 160, 50)
            c0 = torch.zeros(1, 160, 50)
            agent.acnet.eval()
            for cur_step in range(3):
                if cur_step == 0:
                    img_test = env.get_data_test()
                    env_steps = np.zeros(len(img_test), dtype=int)
                else:
                    img_test = env.get_test_imgs()
                    env_steps = env.get_test_steps()
                obss = torch.FloatTensor(img_test.transpose(0, 3, 1, 2)).to(device)

                action_in_test = np.zeros([len(img_test), 12])
                if cur_step>0:
                    for k in range(len(img_test)):
                        if pre_action_test[k] < 12:
                            action_in_test[k, pre_action_test[k]] = 1.
                policy_test, _, ht, ct = agent.acnet(obss, torch.FloatTensor(action_in_test), 160, 1, h0, c0, test=True, off=True)
                h0 = ht
                c0 = ct
                action_test = policy_test.multinomial(1)
                action_test = action_test.cuda().data.cpu().detach().numpy().squeeze()
                if cur_step > 0:
                    pre_action_test = pre_action_test.squeeze()
                    action_test[pre_action_test == 12] = 12
                action_test[env_steps == 3] = 12
                action_test = action_test.reshape((160,1))
                nums_12 = 0
                for i in range(160):
                    if action_test[i][0] == 12:
                        nums_12 += 1
                print("$$$$$$$$")
                print("cur_step:", cur_step, " nums12:", 100 * nums_12 / 160)
                print("$$$$$$$$")
                pre_action_test = action_test
                reward_test, psnr_test, base_psnr, reward_all = env.step_test(action_test, step=cur_step)

                for i in range(5):
                    print(policy_test[i])
                reward_sum += reward_test
                reward_test_vec.append(reward_test)
                psnr_test_vec.append(psnr_test)
                reward_str += 'reward' + str(cur_step + 1) + ': %.4f, '
                psnr_str += 'psnr' + str(cur_step + 1) + ': %.4f, '
            agent.acnet.train()
            # test_reward = np.mean(ep_reward)
            # print("frame_idx:", i,"avg_loss:", total_loss/frame_idx,"avg_reward:", avg_reward/frame_idx, "avg_ep_reward:", test_reward)
            ######
            reward_str += 'reward_sum: %.4f, '
            # if reward_sum > 4.1:
            #     self.learning_rate = 0.0001
            reward_test_vec.append(reward_sum)
            if reward_sum > max_reward and i_epoch % 500 == 0:
                max_reward = reward_sum
                torch.save(agent.acnet, 'ppo_model_test3.pkl')
                print("update_model,**************")
            # sum_reward = reward_sum
            print_str = reward_str + psnr_str + 'base_psnr: %.4f'
            print("clip_param:", agent.clip_param_now)
            print(print_str % tuple(reward_test_vec + psnr_test_vec + [base_psnr]))
            # self.scheduler.step(reward_sum)
            sum_reward.append(reward_sum)
            plt.ion()
            fig = plt.figure(figsize=(8, 6))
            fig.add_subplot(221)
            plt.plot(sum_reward, color="black",label='test_reward')
            fig.add_subplot(222)
            plt.plot(total_loss, color="c", label='total_loss')
            fig.add_subplot(223)
            plt.plot(actor_loss, color="red", label='actor_loss')
            fig.add_subplot(224)
            plt.plot(critic_loss, color="orange", label='critic_loss')
            plt.pause(1.5)
            plt.close()

if __name__ == '__main__':
    main()
    print("end")