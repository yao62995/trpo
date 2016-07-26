#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao62995@gmail.com

import re
import gym
import scipy.signal

from common import *

tf.app.flags.DEFINE_string("game", default_value="BipedalWalker-v2", docstring="game name")
tf.app.flags.DEFINE_string("train_dir", "./trpo_models/", "gym environment name")
tf.app.flags.DEFINE_integer("gpu", default_value=0, docstring="gpu card id")
tf.app.flags.DEFINE_float("value_lr", default_value=5e-4, docstring="learning rate of value network")

tf.app.flags.DEFINE_float("gamma", default_value=0.99, docstring="discount of returns")
tf.app.flags.DEFINE_float("max_kl", default_value=0.01, docstring="max value of kl divergence")
tf.app.flags.DEFINE_integer("rollout_timesteps", default_value=1e4, docstring="number of rollout timesteps")
tf.app.flags.DEFINE_integer("timestep_limit", default_value=0, docstring="game rollout timestep limit")

flags = tf.app.flags.FLAGS


class ControlEnv(object):
    def __init__(self, env):
        self.env = env
        self.state = None

    @property
    def state_dim(self):
        return self.env.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.env.action_space.shape[0]

    def reset_env(self):
        self.state = self.env.reset()
        return self.state

    def forward_action(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.state = obs
        return obs, reward, done

    def timestep_limit(self):
        return self.env.spec.timestep_limit


class PolicyNet(object):
    def __init__(self, scope, state_dim, action_dim):
        self.action_dim = action_dim
        with tf.device("/gpu:%d" % flags.gpu):
            with tf.variable_scope(scope):
                # placeholder
                self.state = tf.placeholder(tf.float32, shape=[None, state_dim], name="state")
                self.action = tf.placeholder(tf.float32, shape=[None, action_dim], name="action")
                self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")
                self.old_dist_n = tf.placeholder(tf.float32, shape=[None, action_dim * 2], name="old_dist_n")
                # mlp
                h1, self.w1, self.b1 = full_connect(self.state, (state_dim, 128), "fc1", with_param=True)
                h2, self.w2, self.b2 = full_connect(h1, (128, 128), "fc2", with_param=True)
                h3, self.w3, self.b3 = full_connect(h2, (128, action_dim), "fc3", with_param=True, activate=None)
                self.logits = tf.identity(h3, name="policy_out")
                self.dist_n = tf.concat(1, [h3, tf.exp(h3) + 1], name="dist_n")
                # log likelihood
                old_logp_n = self.log_likelihood(self.action, self.old_dist_n)
                logp_n = self.log_likelihood(self.action, self.dist_n)
                # loss
                self.surrogate_loss = - tf.reduce_mean(tf.exp(logp_n - old_logp_n) * self.advantage, name="sur_loss")
                self.entropy_loss = tf.reduce_mean(self.entropy(self.dist_n), name="entropy_loss")
                self.kl_loss = tf.reduce_mean(self.kl_divergence(self.old_dist_n, self.dist_n), name="kl_loss")
                self.kl_first_fixed = tf.reduce_mean(self.kl_divergence(tf.stop_gradient(self.dist_n), self.dist_n),
                                                     name="kl_first_fixed")
                kl_grads = tf.gradients(self.kl_first_fixed, self.get_vars(), name="kl_ff_grads")
                pi_grads = tf.gradients(self.surrogate_loss, self.get_vars(), name="policy_grads")
                self.pi_grads = self.get_flat(pi_grads)
                # fisher vector product  (gvp: gradient vector product, fvp: fisher vector product)
                self.flat_theta = tf.placeholder(tf.float32, shape=[None], name="flat_tangent")
                tangent = self.set_from_flat(self.flat_theta)
                gvp = [tf.reduce_sum(tg * kg) for tg, kg in zip(tangent, kl_grads)]
                self.fvp = self.get_flat(gvp)
                # gf: get flat,  sff: set from flat
                self.gf_vars = self.get_flat(self.get_vars())
                sff_vars = [tf.assign(var, t) for t, var in zip(self.set_from_flat(self.flat_theta), self.get_vars())]
                self.sff_vars_op = tf.group(*sff_vars)
                # summary
                summaries = list()
                summaries.append(tf.scalar_summary("surrogate_loss", self.surrogate_loss, name="sur_loss"))
                summaries.append(tf.scalar_summary("entropy_loss", self.surrogate_loss, name="ent_loss"))
                summaries.append(tf.scalar_summary("kl_loss", self.kl_loss, name="kl_loss"))
                self.summary_op = tf.merge_summary(summaries, name="policy_summary_op")
                # set global step
                self.global_step = tf.get_variable("value_net_global_step", shape=[],
                                                   initializer=tf.constant_initializer(0), trainable=False)
                self.augment_flag = tf.placeholder(tf.bool, shape=[], name="augment_global_step")
                self.augment_step_op = tf.assign(self.global_step,
                                                 tf.select(self.augment_flag, self.global_step + 1, self.global_step))

    def train(self, paths, sess, summary_writer):
        feed_dict = {
            self.state: paths["states"],
            self.action: paths["actions"],
            self.advantage: paths["advant"],
            self.old_dist_n: paths["dist_n"]
        }

        def fisher_vector_product(g):
            feed_dict[self.flat_theta] = g
            return sess.run(self.fvp, feed_dict=feed_dict)

        def compute_loss(theta):
            sess.run(self.sff_vars_op, feed_dict={self.flat_theta: theta})
            return sess.run(self.surrogate_loss, feed_dict=feed_dict)

        # policy gradients
        pi_grads = sess.run(self.pi_grads, feed_dict=feed_dict)
        # flatten vars
        theta_prev = sess.run(self.gf_vars)
        # conjugate gradient
        step_dir = self.conjugate_gradient(fisher_vector_product, -pi_grads)
        # lagrange multiplier
        lm = np.sqrt(0.5 * step_dir.dot(fisher_vector_product(step_dir)) / flags.max_kl)
        full_step = step_dir / lm
        expected_improve_rate = -pi_grads.dot(step_dir) / lm
        theta_new = self.line_search(compute_loss, theta_prev, full_step, expected_improve_rate)
        sess.run([self.sff_vars_op, self.augment_step_op], feed_dict={self.flat_theta: theta_new,
                                                                      self.augment_flag: True})
        summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=self.global_step)

    def get_flat(self, grads):
        flat_grads = []
        for grad, var in zip(grads, self.get_vars()):
            flat_grads.append(tf.reshape(grad, shape=[np.prod(var.get_shape().as_list())]))
        theta = tf.concat(0, flat_grads)
        return theta

    def set_from_flat(self, theta):
        grads = []
        start_size = 0
        for var in self.get_vars():
            var_size = np.prod(var.get_shape().as_list())
            grad = tf.reshape(theta[start_size: (start_size + var_size)], shape=var.get_shape().as_list())
            grads.append(grad)
            start_size += var_size
        return grads

    def conjugate_gradient(self, fvp_func, b, cg_iters=10, residual_tol=1e-10):
        p = b.copy()
        r = b.copy()
        x = np.zeros_like(b)
        rdotr = r.dot(r)
        for i in xrange(cg_iters):
            z = fvp_func(p)
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z
            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < residual_tol:
                break
        return x

    def line_search(self, loss_func, theta, full_step, expected_improve_rate):
        accept_ratio = .1
        max_backtracks = 10
        sff_val = loss_func(theta)
        for (_n_backtracks, step_frac) in enumerate(.5 ** np.arange(max_backtracks)):
            theta_new = theta + step_frac * full_step
            new_sff_val = loss_func(theta_new)
            actual_improve = sff_val - new_sff_val
            expected_improve = expected_improve_rate * step_frac
            ratio = actual_improve / expected_improve
            if ratio > accept_ratio and actual_improve > 0:
                return theta_new
        return theta

    def entropy(self, dist_prob):
        std = dist_prob[:, self.action_dim:]
        ent_const = tf.constant(0.5 * np.log(2 * np.pi * np.e) * self.action_dim, name="entropy_const", shape=[])
        return tf.reduce_sum(tf.log(std) + ent_const, reduction_indices=1)

    def log_likelihood(self, act, dist_prob):
        mean, std = dist_prob[:, :self.action_dim], dist_prob[:, self.action_dim:]
        log_const = 0.5 * np.log(2 * np.pi) * self.action_dim
        return -0.5 * tf.reduce_sum(tf.square(act - mean / std), reduction_indices=1) \
               - tf.reduce_sum(tf.log(std), reduction_indices=1) \
               - log_const

    def kl_divergence(self, old_dist_n, dist_n):
        mean_0, std_0 = old_dist_n[:, :self.action_dim]
        mean_1, std_1 = dist_n[:, self.action_dim:]
        return tf.reduce_sum(tf.log(std_1 / std_0), reduction_indices=1) + \
               tf.reduce_sum(tf.div(tf.square(std_0) + tf.square(mean_0 - mean_1), 2.0 * tf.square(std_1)),
                             reduction_indices=1) - \
               0.5 * self.action_dim

    def get_policy(self, sess, states):
        return sess.run(self.logits, feed_dict={self.state: states})

    def get_dist_n(self, sess, states):
        return sess.run(self.dist_n, feed_dict={self.state: states})

    def get_sample(self, sess, states):
        dist_n = self.get_dist_n(sess, states)
        mean_nd = dist_n[:, :self.action_dim]
        std_nd = dist_n[:, self.action_dim:]
        action = np.random.randn(len(states), self.action.shape[1]) * std_nd + mean_nd
        return action, dist_n

    def get_vars(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]


class ValueNet(object):
    def __init__(self, scope, state_dim, timestep_limit):
        self.timestep_limit = timestep_limit
        with tf.device("/gpu:%d" % flags.gpu):
            with tf.variable_scope(scope) as scope:
                # state input shape(None, state_dim + 1), the extra one is percent of time step
                self.state = tf.placeholder(tf.float32, shape=[None, state_dim + 1], name="state")
                self.target_q = tf.placeholder(tf.float32, shape=[None], name="target_q")
                self.global_step = tf.get_variable("value_net_global_step", shape=[],
                                                   initializer=tf.constant_initializer(0), trainable=False)
                # mlp
                h1, self.w1, self.b1 = full_connect(self.state, (state_dim + 1, 128), "fc1", with_param=True,
                                                    weight_decay=0.01)
                h2, self.w2, self.b2 = full_connect(h1, (128, 128), "fc2", with_param=True,
                                                    weight_decay=0.01)
                h3, self.w3, self.b3 = full_connect(h2, (128, 1), "fc3", with_param=True, activate=None,
                                                    weight_decay=0.01)
                self.logits = tf.reshape(h3, shape=[-1], name="value_out")
                # losses
                l2_loss = tf.add_n(tf.get_collection("losses", scope=scope), name="l2_loss")
                self.loss = tf.reduce_sum(tf.square(self.logits - self.target_q)) + l2_loss
                self.opt = tf.train.GradientDescentOptimizer(flags.value_lr).minimize(self.loss,
                                                                                      global_step=self.global_step)
                # summary op
                value_summary = tf.scalar_summary("value_loss", self.loss)
                self.summary_op = tf.merge_summary([value_summary])

    def add_extra_dim(self, states):
        return np.concatenate([states, np.arange(len(states)).reshape(-1, 1) / float(self.timestep_limit)], axis=1)

    def get_value(self, sess, states):
        states = self.add_extra_dim(states)
        return sess.run(self.logits, feed_dict={self.state: states})

    def get_vars(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def train(self, paths, session, summary_writer, n_iter=20):
        for i in xrange(n_iter):
            fetches = [self.opt]
            if i % 10 == 0:
                fetches.append(self.summary_op)
            res = session.run(fetches, feed_dict={self.state: paths["states"], self.target_q: paths["returns"]})
            if i % 10 == 0:
                summary_str = res[1]
                summary_writer.add_summary(summary_str, global_step=self.global_step)


class TRPO(object):
    """paper: Trust Region Policy Optimization - arXiv.org"""

    def __init__(self):
        self.env = ControlEnv(gym.make(flags.game))
        self.timestep_limit = flags.timestep_limit or self.env.timestep_limit()
        # basic network
        self.policy_net = PolicyNet("policy_net", self.env.state_dim, self.env.action_dim)
        self.value_net = ValueNet("value_net", self.env.action_dim, self.timestep_limit)
        # session
        self.sess = tf.Session(graph=tf.get_default_graph(),
                               config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        self.sess.run(tf.initialize_all_variables())
        self.summary_writer = tf.train.SummaryWriter(flags.train_dir, graph_def=self.sess.graph)
        # saver
        saved_var_list = self.policy_net.get_vars() + self.value_net.get_vars() + \
                         [self.policy_net.global_step, self.value_net.global_step]
        self.saver = tf.train.Saver(max_to_keep=3, var_list=saved_var_list)
        restore_model(self.sess, flags.train_dir, self.saver)

    def rollout(self):
        paths = []
        rollout_time_step = 0
        while rollout_time_step < flags.rollout_timesteps:
            self.env.reset_env()
            terminal = False
            time_step = 0
            path = {"obs": [], "action": [], "dist_n": [], "reward": [], "done": []}
            while not terminal and time_step < self.timestep_limit:
                state = self.env.state
                action, dist_n = self.policy_net.get_sample(self.sess, [state])[0]
                state_n, reward, terminal = self.env.forward_action(action)
                path["obs"].append(state)
                path["action"].append(action)
                path["dist_n"].append(dist_n)
                path["reward"].append(reward)
                path["done"].append(terminal)
                time_step += 1
            paths.append(path)
            rollout_time_step += time_step
        return paths

    @staticmethod
    def discount(x):
        return scipy.signal.lfilter([1], [1, -flags.gamma], x[::-1], axis=0)[::-1]

    def compute_advantage(self, paths):
        for path in paths:
            value_base = self.value_net.get_value(self.sess, path["obs"])
            if path["done"][-1]:
                path["reward"].append(0)
            else:
                path["reward"].append(value_base[-1])
            path["returns"] = self.discount(path["reward"])[:-1]
            path["advant"] = path["returns"] - value_base
        # concat paths
        states = np.concatenate([self.value_net.add_extra_dim(path["obs"]) for path in paths], axis=0)
        returns = np.concatenate([path["returns"] for path in paths], axis=0)
        action = np.concatenate([path["action"] for path in paths], axis=0)
        dist_n = np.concatenate([path["dist_n"] for path in paths], axis=0)
        advant = np.concatenate([path["advant"] for path in paths], axis=0)
        del paths
        paths = {"states": states, "returns": returns, "actions": action, "dist_n": dist_n, "advant": advant}
        return paths

    def train(self):
        while True:
            # rollout path
            paths = self.rollout()
            # compute advantage
            paths = self.compute_advantage(paths)
            # fit value network
            self.value_net.train(paths, self.sess, self.summary_writer)
            # policy gradients
            self.policy_net.train(paths, self.sess, self.summary_writer)


def main(_):
    # mkdir
    if not os.path.isdir(flags.train_dir):
        os.makedirs(flags.train_dir)
    # remove old tfevents files
    for f in os.listdir(flags.train_dir):
        if re.search(".*tfevents.*", f):
            os.remove(os.path.join(flags.train_dir, f))
    # model
    model = TRPO()
    model.train()


if __name__ == "__main__":
    tf.app.run()
