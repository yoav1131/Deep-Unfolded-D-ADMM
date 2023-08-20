import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='exp for 5 users new',
                        help="the name of the current experiment")
    parser.add_argument('--eval', action='store_true',
                        help="weather to perform inference of training")

    # data arguments
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist'],
                        help="dataset to use (mnist or cifar)")
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help="trainset batch size")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help="testset batch size")

    # federated arguments
    parser.add_argument('--model', type=str, default='linear',
                        choices=['linear', 'mlp'],
                        help="model to use linear")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users participating in the federated learning")
    parser.add_argument('--local_epochs', type=int, default=20,
                        help="number of local epochs")
    parser.add_argument('--local_iterations', type=int, default=1,
                        help="number of local iterations instead of local epoch")
    parser.add_argument('--global_epochs', type=int, default=1000,
                        help="number of global epochs")
    parser.add_argument('--aggregation_method', default='FedAvg',
                        choices=['FedAvg'],
                        help="centralized or federated learning")
    parser.add_argument('--threshold', type=float, default=None,
                        help="zero the weight if users values summation is beneath the threshold")




    # robustness arguments
    parser.add_argument('--malicious', default=None,
                        choices=[None],
                        help="select malicious users manipulation type")
    parser.add_argument('--malicious_users_percent', type=float, default=0.1,
                        help="select the percentage of malicious users")


    # learning arguments
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="learning rate is 0.0001 for sgd and 0.001  for adam")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="momentum")
    parser.add_argument('--lr_scheduler', action='store_true',
                        help="reduce the learning rat when val_acc has stopped improving (increasing)")
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help="device to use (gpu or cpu)")
    parser.add_argument('--seed', type=float, default=1234,
                        help="manual seed for reproducibility")

    args = parser.parse_args()
    return args
