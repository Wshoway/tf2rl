import roboschool,gym

from tf2rl.trainer.apex_trainer import ApeXTrainer
from tf2rl.algos.ddpg import DDPG


if __name__ == '__main__':
    parser = ApeXTrainer.get_argument()
    parser.add_argument('--env-name', type=str, default="RoboschoolAnt-v1")
    args = parser.parse_args()

    def env_fn():
        return gym.make(args.env_name)

    def policy_fn(env, name, memory_capacity=1e6, gpu=-1):
        return DDPG(
            state_dim=env.observation_space.high.size,
            action_dim=env.action_space.high.size,
            gpu=gpu,
            name=name,
            batch_size=100,
            memory_capacity=memory_capacity)

    apex_trainer = ApeXTrainer(args, env_fn, policy_fn)
    apex_trainer()
