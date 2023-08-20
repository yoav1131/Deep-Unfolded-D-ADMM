import torch
import torch.optim as optim
import copy

def federated_setup(global_model, train_data, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    indexes = torch.randperm(len(train_data))
    user_data_len = 5  # math.floor(len(train_data) / args.max_num_users)
    local_models = {}
    for user_idx in range(args.num_users):
        user = {'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data,
                                    indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]),
            batch_size=args.train_batch_size, shuffle=True),
            'model': copy.deepcopy(global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        if args.lr_scheduler:
            user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user['opt'], patience=10, factor=0.1, verbose=True)
        local_models[user_idx] = user
    return local_models


def distribute_model(local_models, global_model):
    for user_idx in range(len(local_models)):
        local_models[user_idx]['model'].load_state_dict(copy.deepcopy(global_model.state_dict()))


mean = lambda x: sum(x) / len(x)


class FedAvg:  # Compression Privacy class
    def __init__(self, args):
        self.compression = None
        self.privacy = None
        self.threshold = args.threshold
        self.device = args.device
        self.malicious = args.malicious

    def __call__(self, input, malicious=None):
        input = self.compression(input, malicious) if self.compression is not None else input
        input = self.privacy(input) if self.privacy is not None else input
        return input


    def fed_avg(self, local_models, global_model):
        state_dict = copy.deepcopy(global_model.state_dict())
        # SNR_layers = []
        for key in state_dict.keys():
            local_weights_average = torch.zeros_like(state_dict[key].view(-1))
            local_weights_orig_average = torch.zeros_like(state_dict[key].view(-1))

            for user_idx in range(0, len(local_models)):
                local_weights_orig = local_models[user_idx]['model'].state_dict()[key].view(-1) - state_dict[key].view(-1)
                local_weights = self(local_weights_orig)
                local_weights_average += local_weights

            # local_weights_orig_average = local_weights_orig_average / len(local_models)
            local_weights_average = local_weights_average / len(local_models)
            state_dict[key] += local_weights_average.reshape(state_dict[key].shape).to(state_dict[key].dtype)

        global_model.load_state_dict(copy.deepcopy(state_dict))
