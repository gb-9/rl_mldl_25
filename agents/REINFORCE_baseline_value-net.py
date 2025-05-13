import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        self.fc1_critic  = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic  = torch.nn.Linear(self.hidden,  self.hidden)
        self.fc3_value   = torch.nn.Linear(self.hidden,  1) #scalar value


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        x_v = self.tanh(self.fc1_critic(x))
        x_v = self.tanh(self.fc2_critic(x_v))
        value = self.fc3_value(x_v).squeeze(-1)

        
        return normal_dist, value


class Agent(object):
    def __init__(self, policy, device='cpu', entropy_coef=0.01, max_grad_norm=None):
        self.train_device = device
        self.policy = policy.to(self.train_device)

        actor_params  = [p for n, p in self.policy.named_parameters()
                         if "actor" in n or "sigma" in n]
        critic_params = [p for n, p in self.policy.named_parameters()
                         if "critic" in n or "value"  in n]

        self.policy_optimizer = torch.optim.Adam(actor_params,  lr=1e-3)
        self.value_optimizer  = torch.optim.Adam(critic_params, lr=1e-3)

        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.gamma = 0.99
        self.states, self.next_states = [], []
        self.action_log_probs, self.rewards, self.done = [], [], []
        self.entropies = []


    def update_policy(self):
        states   = torch.stack(self.states).to(self.train_device)
        log_p    = torch.stack(self.action_log_probs).to(self.train_device)
        entrop   = torch.stack(self.entropies).to(self.train_device)
        rewards  = torch.stack(self.rewards).to(self.train_device).squeeze(-1)

        returns = discount_rewards(rewards, self.gamma)

        _, state_values = self.policy(states)        
        value_error     = returns - state_values 

        T = returns.size(0)
        discounts = self.gamma ** torch.arange(T, dtype=returns.dtype,
                                            device=self.train_device)
        
        policy_loss = -(discounts * value_error.detach() * log_p).mean()

        entropy_loss = -entrop.mean()
        actor_loss   = policy_loss + self.entropy_coef * entropy_loss

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        value_loss = 0.5 * value_error.pow(2).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

        self.states.clear()
        self.next_states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.done.clear()
        self.entropies.clear()


        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()
            entropy  = normal_dist.entropy().sum()

            return action, (action_log_prob, entropy)


    def store_outcome(self, state, next_state, log_ent_tuple, reward, done):
        action_log_prob, entropy = log_ent_tuple

        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.entropies.append(entropy)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
