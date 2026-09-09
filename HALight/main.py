import argparse
import torch
from runner import OnPolicyBaseRunner

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prs.add_argument(
        "-flow_dir",
        dest="flow_dir",
        type=str,
        default="../../flows/Binjiang_high",
        help="Flow folder used for both training and testing. Training randomly samples one json flow file per episode; after training, testing traverses all json flow files in this folder.",
    )
    prs.add_argument(
        "-net",
        dest="net",
        type=str,
        default="../../Nets/Binjiang/Binjiang.json",
    )
    prs.add_argument("-dir", dest="dir", type=str, default="./", required=False)
    prs.add_argument("-PATH_TO_WORK_DIRECTORY", dest="PATH_TO_WORK_DIRECTORY", type=str, default="./", required=False)
    prs.add_argument("-lr", dest="lr", type=float, default=0.00005, required=False, help="learning rate.\n")
    prs.add_argument("-critic_lr", dest="critic_lr", type=float, default=0.0005, required=False, help="learning rate.\n")
    prs.add_argument("-obs_drop_prob", dest="obs_drop_prob", type=float, default=0.0, required=False,help="observation dropout probability.\n")
    prs.add_argument("-gamma", dest="gamma", type=float, default=0.95, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-episode", dest="episode", type=int, default=10000, help="Number of episodes.\n")
    prs.add_argument("-seed", dest="seed", type=int, default=42, help="The seed of the experiment.\n")
    prs.add_argument("-delta_time", dest="delta_time", type=int, default=10, help="The time of a step.\n")
    prs.add_argument("-infor_dim", dest="infor_dim", type=int, default=8, help="The dim of information.\n")
    prs.add_argument("-ht_dim", dest="ht_dim", type=int, default=16, help="The dim of ht.\n")
    prs.add_argument("-attn_dim", dest="attn_dim", type=int, default=16, help="The dim of attention.\n")
    prs.add_argument("-episode_length", dest="episode_length", type=int, default=360, help="max steps.\n")
    prs.add_argument("-use_proper_time_limits", dest="use_proper_time_limits", type=bool, default=True)
    prs.add_argument("-save_replay", dest="save_replay", type=bool, default=False)
    prs.add_argument("-hidden_sizes", dest="hidden_sizes", type=int, default=64)
    prs.add_argument("-critic_hidden_sizes", dest="critic_hidden_sizes", type=int, default=128)
    prs.add_argument("-ppo_epoch", dest="ppo_epoch", type=int, default=10)
    prs.add_argument("-critic_epoch", dest="critic_epoch", type=int, default=10)
    prs.add_argument("-use_clipped_value_loss", dest="use_clipped_value_loss", type=bool, default=False)
    prs.add_argument("-clip_param", dest="clip_param", type=float, default=0.2)
    prs.add_argument("-actor_num_mini_batch", dest="actor_num_mini_batch", type=int, default=1)
    prs.add_argument("-critic_num_mini_batch", dest="critic_num_mini_batch", type=int, default=1)
    prs.add_argument("-entropy_coef", dest="entropy_coef", type=float, default=0.01)
    prs.add_argument("-value_loss_coef", dest="value_loss_coef", type=float, default=1)
    prs.add_argument("-use_max_grad_norm", dest="use_max_grad_norm", type=bool, default=False)
    prs.add_argument("-max_grad_norm", dest="max_grad_norm", type=float, default=10)
    prs.add_argument("-use_gae", dest="use_gae", type=bool, default=True)
    prs.add_argument("-gae_lambda", dest="gae_lambda", type=float, default=0.9)
    prs.add_argument("-use_huber_loss", dest="use_huber_loss", type=bool, default=True)
    prs.add_argument("-huber_delta", dest="huber_delta", type=float, default=10)
    prs.add_argument("-fixed_order", dest="fixed_order", type=bool, default=False)
    prs.add_argument("-opti_eps", dest="opti_eps", type=float, default=0.00001)
    prs.add_argument("-opti_delta", dest="opti_delta", type=float, default=0.00001)
    prs.add_argument("-use_valuenorm", dest="use_valuenorm", type=bool, default=False)
    prs.add_argument("-use_critic_lr_decay", dest="use_critic_lr_decay", type=bool, default=False) #得加上这部分的代码

    args = prs.parse_args()

    """
    开始训练
    """
    runner = OnPolicyBaseRunner(args)
    runner.run()



