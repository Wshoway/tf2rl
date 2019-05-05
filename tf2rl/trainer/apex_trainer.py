import time
import numpy as np
import tensorflow as tf
import sys

import multiprocessing
from multiprocessing import Process, Queue, Value, Event, Lock
from multiprocessing.managers import SyncManager

from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from tf2rl.trainer.trainer import Trainer
from tf2rl.misc.target_update_ops import update_target_variables

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class ApeXTrainer(Trainer):
    def __init__(self, args, env_fn, policy_fn):
        learner_env = env_fn()
        learner_policy = policy_fn(learner_env, name="Learner")
        super().__init__(policy=learner_policy, env=learner_env, args=args)
        self._env_fn = env_fn
        self._policy_fn = policy_fn

    def __call__(self):
        dummy_env = self._env_fn()
        # Manager to share PER between a learner and explorers
        SyncManager.register('PrioritizedReplayBuffer', PrioritizedReplayBuffer)
        manager = SyncManager()
        manager.start()
        global_rb = manager.PrioritizedReplayBuffer(
            obs_dim=dummy_env.observation_space.low.size,
            act_dim=dummy_env.action_space.low.size,
            size=1e6)

        # queues to share network parameters between a learner and explorers
        queues = [Queue() for _ in range(self._n_explorer)]

        # Event object to share training status. if event is set True, all exolorers stop sampling transitions
        is_training_done = Event()

        # Lock
        lock = manager.Lock()

        # Shared memory objects to count number of samples and applied gradients
        trained_steps = Value('i', 0)
        n_transition = Value('i', 0)

        tasks = []
        # Add explorers
        for i in range(self._n_explorer):
            tasks.append(Process(
                target=explorer,
                args=[global_rb, queues[i], trained_steps, n_transition,
                      is_training_done, lock, self._env_fn, self._policy_fn,
                      self._local_buffer_size, self._episode_max_steps]))

        # Add learner
        tasks.append(Process(
            target=learner,
            args=[global_rb, trained_steps, is_training_done, lock,
                  self._env, self._policy, self.writer, self._n_apply_grad,
                  self._param_update_freq, *queues]))

        for task in tasks:
            task.start()
        for task in tasks:
            task.join()

    def _set_from_args(self, args):
        super()._set_from_args(args)
        if args.n_explorer is None:
            self._n_explorer = multiprocessing.cpu_count() - 1
        else:
            self._n_explorer = args.n_explorer
        assert self._n_explorer > 0, "[error] number of explorers must be positive integer"
        self._param_update_freq = args.param_update_freq
        self._n_apply_grad = args.n_apply_grad
        self._local_buffer_size = args.local_buffer_size

    @staticmethod
    def get_argument(parser=None):
        parser = Trainer.get_argument(parser)
        parser.add_argument('--param-update-freq', type=int, default=50,
                            help='Frequency to update parameter')
        parser.add_argument('--n-explorer', type=int, default=None,
                            help='Number of explorers to distribute')
        parser.add_argument('--n-apply-grad', type=int, default=int(1e6),
                            help="Number of times to apply gradients")
        parser.add_argument('--local-buffer-size', type=int, default=1e4,
                            help='size of local replay buffer for explorer')
        return parser


def print_out(data):
    print(data)
    sys.stdout.flush()


def explorer(global_rb, queue, trained_steps, n_transition,
             is_training_done, lock, env_fn, policy_fn,
             local_buffer_size, episode_max_steps):
    """
    Collect transitions and store them to prioritized replay buffer.
    Args:
        global_rb:
            Prioritized replay buffer sharing with multiple explorers and only one learner.
            This object is shared over processes, so it must be locked when trying to
            operate something with `lock` object.
        queue:
            A FIFO shared with the learner to get latest network parameters.
            This is process safe, so you don't need to lock process when use this.
        trained_steps:
            Number of steps to apply gradients.
        n_transition:
            Number of collected transitions.
        is_training_done:
            multiprocessing.Event object to share the status of training.
        lock:
            multiprocessing.Lock to lock other processes. You must release after process is done.
        env_fn:
            Method object to generate an environment.
        policy_fn:
            Method object to generate an explorer.
        buffer_size:
            Size of local buffer. If it is filled with transitions, add them to `global_rb`
        episode_max_steps:
            Maximum number of steps of an episode.
    """
    print_out("env")
    env = env_fn()
    print_out("obs: {} act: {}".format(
        env.observation_space.high.size,
        env.action_space.high.size
    ))
    print_out("policy {}".format(policy_fn))
    policy = policy_fn(env, "Explorer")
    print_out("local_rb")
    local_rb = ReplayBuffer(obs_dim=env.observation_space.low.size,
                            act_dim=env.action_space.low.size,
                            size=episode_max_steps)

    s = env.reset()
    episode_steps = 0
    total_reward = 0.
    total_rewards = []
    start = time.time()
    sample_at_start = 0

    while not is_training_done.is_set():
        print(n_transition.value)
        sys.stdout.flush()
        # Periodically copy weights of explorer
        if not queue.empty():
            actor_weights, critic_weights, critic_target_weights = queue.get()
            update_target_variables(policy.actor.weights, actor_weights, tau=1.)
            update_target_variables(policy.critic.weights, critic_weights, tau=1.)
            update_target_variables(policy.critic_target.weights, critic_target_weights, tau=1.)

        n_transition.value += 1
        episode_steps += 1
        a = policy.get_action(s)
        s_, r, done, _ = env.step(a)
        done_flag = done
        if episode_steps == env._max_episode_steps:
            done_flag = False
        total_reward += r
        local_rb.add(s, a, r, s_, done_flag)

        s = s_
        if done or episode_steps == episode_max_steps:
            s = env.reset()
            total_rewards.append(total_reward)
            total_reward = 0
            episode_steps = 0

        # Add collected experiences to global replay buffer
        if local_rb.get_stored_size() == local_buffer_size - 1:
            temp_n_transition = n_transition.value
            samples = local_rb.sample(local_rb.get_stored_size())
            states, next_states, actions, rewards, done = samples["obs"], samples["next_obs"], samples["act"], samples["rew"], samples["done"]
            done = np.array(done, dtype=np.float64)
            td_errors = policy.compute_td_error(
                states, actions, next_states, rewards, done)
            print("Grad: {0: 6d}\tSamples: {1: 7d}\tTDErr: {2:.5f}\tAveEpiRew: {3:.3f}\tFPS: {4:.2f}".format(
                trained_steps.value, n_transition.value, np.average(np.abs(td_errors).flatten()),
                sum(total_rewards) / len(total_rewards), (temp_n_transition - sample_at_start) / (time.time() - start)))
            sys.stdout.flush()
            total_rewards = []
            lock.acquire()
            global_rb.add(
                states, actions, rewards, next_states, done,
                priorities=np.abs(td_errors)+1e-6)
            lock.release()
            local_rb.clear()
            start = time.time()
            sample_at_start = n_transition.value


def learner(global_rb, trained_steps, is_training_done,
            lock, env, policy, writer, n_training, update_freq, *queues):
    """
    Collect transitions and store them to prioritized replay buffer.
    Args:
        global_rb:
            Prioritized replay buffer sharing with multiple explorers and only one learner.
            This object is shared over processes, so it must be locked when trying to
            operate something with `lock` object.
        trained_steps:
            Number of times to apply gradients.
        is_training_done:
            multiprocessing.Event object to share if training is done or not.
        lock:
            multiprocessing.Lock to lock other processes.
            It must be released after process is done.
        env_fn:
            Method object to generate an environment.
        policy_fn:
            Method object to generate an explorer.
        n_training:
            Maximum number of times to apply gradients. If number of applying gradients
            is over this value, training will be done by setting `is_training_done` to `True`
        update_freq:
            Frequency to update parameters, i.e., put network parameters to `queues`
        queues:
            FIFOs shared with explorers to send latest network parameters.
    """
    # env = env_fn()
    # policy = policy_fn(env, "Learner", global_rb.get_buffer_size())
    update_step = update_freq

    total_steps = tf.train.create_global_step()

    # Wait until explorers collect transitions
    while not is_training_done.is_set() and global_rb.get_stored_size() == 0:
        continue

    start_time = time.time()
    while not is_training_done.is_set():
        with tf.contrib.summary.record_summaries_every_n_global_steps(1000):
            trained_steps.value += 1
            total_steps.assign(trained_steps.value)
            lock.acquire()
            samples = global_rb.sample(policy.batch_size)
            with tf.contrib.summary.always_record_summaries():
                td_error = policy.train(
                    samples["obs"], samples["act"], samples["next_obs"],
                    samples["rew"], np.array(samples["done"], dtype=np.float64),
                    samples["weights"])
                writer.flush()
            global_rb.update_priorities(samples["indexes"], np.abs(td_error) + 1e-6)
            lock.release()

            # Put updated weights to queue
            if trained_steps.value > update_step:
                weights = []
                weights.append(policy.actor.weights)
                weights.append(policy.critic.weights)
                weights.append(policy.critic_target.weights)
                for queue in queues:
                    queue.put(weights)
                update_step += update_freq
                with tf.contrib.summary.always_record_summaries():
                    fps = update_freq / (time.time() - start_time)
                    tf.contrib.summary.scalar(name="FPS", tensor=fps, family="loss")
                    print("Update weights for explorer. {0:.2f} FPS for GRAD. Learned {1:.2f} steps".format(fps, trained_steps.value))
                    sys.stdout.flush()
                start_time = time.time()

        if trained_steps.value >= n_training:
            is_training_done.set()
