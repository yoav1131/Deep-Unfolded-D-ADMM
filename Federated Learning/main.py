import gc
import sys
from statistics import mean
import time
import torch
from configurations import args_parser
from tqdm import tqdm
import utils
import models
import federated_utils
import numpy as np


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    boardio, textio, best_val_acc, path_best_model = utils.initializations(args)
    textio.cprint(str(args))

    # data
    train_data, test_loader = utils.data(args)
    input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)

    # model
    if args.model == 'mlp':
        global_model = models.FC2Layer(input, output)
    else:
        global_model = models.Linear(input, output)
    # textio.cprint(str(summary(global_model)))
    global_model.to(args.device)

    train_creterion = torch.nn.MSELoss(reduction='mean')
    test_creterion = torch.nn.MSELoss(reduction='mean')

    # learning curve
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    train_acc_list = []

    #  inference
    if args.eval:
        global_model.load_state_dict(torch.load(path_best_model))
        test_acc = utils.test(test_loader, global_model, test_creterion, args.device)
        textio.cprint(f'eval test_acc: {test_acc:.0f}%')
        gc.collect()
        sys.exit()

    local_models = federated_utils.federated_setup(global_model, train_data, args)
    aggregate_models = federated_utils.FedAvg(args).fed_avg

    # SNR_list = []
    for global_epoch in tqdm(range(0, args.global_epochs)):
        federated_utils.distribute_model(local_models, global_model)
        users_loss = []
        users_acc = []

        for user_idx in range(args.num_users):
            user_loss = []
            user_acc = []
            for local_epoch in range(0, args.local_epochs):
                user = local_models[user_idx]
                train_loss, train_acc = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                                                   train_creterion, args.device, args.local_iterations)
                if args.lr_scheduler:
                    user['scheduler'].step(train_loss)
                user_loss.append(train_loss)
                user_acc.append(train_acc)
            users_loss.append(mean(user_loss))
            users_acc.append(mean(user_acc))

        train_loss = mean(users_loss)
        train_acc = mean(users_acc)
        aggregate_models(local_models, global_model)

        val_acc, val_loss = utils.test(val_loader, global_model, test_creterion, args.device)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        boardio.add_scalar('train loss', train_loss, global_epoch)
        boardio.add_scalar('train acc', train_acc, global_epoch)
        boardio.add_scalar('validation loss', val_loss, global_epoch)
        boardio.add_scalar('validation acc', val_acc, global_epoch)
        gc.collect()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model.state_dict(), path_best_model)

        textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss:.2f}, train_acc: {train_acc:.0f}% | '
                      f'val_loss: {val_loss:.2f}, val_acc: {val_acc:.0f}%\n')

    np.save(f'checkpoints/{args.exp_name}/train_loss_list_P={args.num_users}.npy', train_loss_list)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list_P={args.num_users}.npy', val_acc_list)
    np.save(f'checkpoints/{args.exp_name}/train_acc_list_P={args.num_users}.npy', train_acc_list)
    np.save(f'checkpoints/{args.exp_name}/val_loss_list_P={args.num_users}.npy', val_loss_list)
    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.0f} min')
