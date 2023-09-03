import gc
import sys
import time
import torch
from configurations import args_parser
from tqdm import tqdm
import utils
import models
import os
import numpy as np
import DADMM_utils


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    boardio, textio, best_val_loss = utils.initializations(args)
    textio.cprint(str(args))

    # data
    train_data, test_loader = utils.data(args)
    train_loader, val_loader = utils.data_split(train_data, args)

    # Graph creation
    if os.path.isfile(f'graphs data/erods_renyi/graph_data_prob{args.graph_prob}_P={args.P}.npy'):
        with open(f'graphs data/erods_renyi/graph_data_prob{args.graph_prob}_P={args.P}.npy', 'rb') as f:  # If I want to use same graph!!
            neighbors = np.load(f, allow_pickle=True)
            color_partition = np.load(f, allow_pickle=True)
    else:
        net = DADMM_utils.CreateGraph(args)
        neighbors = DADMM_utils.CreateGraph.graph2array(net.net1)
        color_partition = DADMM_utils.CreateGraph.proper_coloring_algorithm(net.net1)
        with open(f'graphs data/erods_renyi/graph_data_prob{args.graph_prob}_P={args.P}.npy', 'wb') as f:  # If I want to use same graph!!
            np.save(f, neighbors, allow_pickle=True)
            np.save(f, color_partition, allow_pickle=True)

    # model hypeparameters
    learn_params = torch.tensor([[args.rho, args.alpha, args.tau, args.eta]] * args.P).unsqueeze(0).repeat([args.max_iter, 1, 1]).to(args.device)

    #  inference
    if args.eval:
        if args.method == 'u-dadmm':  # check if we want to use u-dadmm or dadmm method
            learn_params = torch.load(f'checkpoints/{args.case}/{args.hyper_parameters}/params_P={args.P}_batch={args.max_iter}_prob={args.graph_prob}_size={args.max_iter}.pt')  # load the learned hyperparameters
        global_model = models.DADMMLASSO(learn_params, learn_params.shape[0], args, no_hyp=learn_params)
        global_model.to(args.device)
        test_loss, _ = utils.test(test_loader, global_model, neighbors, color_partition, args)
        with open(f'checkpoints/{args.case}/{args.exp_name}/{args.model}/results/params_P={args.P}_prob={args.graph_prob}_iteration_{args.max_iter}_{args.method}.npy', 'wb') as f:
            np.save(f, np.array(test_loss), allow_pickle=True)
        gc.collect()
        sys.exit()

    # learning curve
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    train_acc_list = []

    # sequential training model
    for ll in range(0, args.max_iter, args.max_iter_seg):
        args.seq_num = ll
        path_best_model = utils.init_best_path(args)
        if ll == 0:
            global_model = models.DADMMLASSO(learn_params[:ll+args.max_iter_seg], ll, args, no_hyp=None)
            global_model.to(args.device)
            for epoch in tqdm(range(0, args.num_epochs)):
                train_loss, _ = utils.train_one_epoch(global_model, train_loader, neighbors, color_partition, args)
                val_loss, _ = utils.test(val_loader, global_model, neighbors, color_partition, args)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)

                boardio.add_scalar('train loss', train_loss, epoch)
                boardio.add_scalar('validation loss', val_loss, epoch)
                gc.collect()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(global_model.state_dict(), path_best_model.format())
                textio.cprint(f'epoch: {epoch} | '
                          f'train_loss: {train_loss:.2f} | '
                          f'valid_loss: {val_loss:.3f}')
            not_learn_params = global_model.state_dict()['hyp']
            torch.save(not_learn_params, f'checkpoints/{args.case}/{args.exp_name}/{args.model}/params_P={args.P}_batch={not_learn_params.shape[0]}_prob={args.graph_prob}_size={ll + args.max_iter_seg}.pt')
        else:
            global_model = models.DADMMLASSO(learn_params[ll:ll + args.max_iter_seg], ll, args, not_learn_params)
            global_model.to(args.device)
            for epoch in tqdm(range(0, args.num_epochs)):
                train_loss, _ = utils.train_one_epoch(global_model, train_loader, neighbors, color_partition, args)
                val_loss, _ = utils.test(val_loader, global_model, neighbors, color_partition, args)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)

                boardio.add_scalar('train loss', train_loss, epoch)
                boardio.add_scalar('validation loss', val_loss, epoch)
                gc.collect()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(global_model.state_dict(), path_best_model.format())
                textio.cprint(f'epoch: {epoch} | '
                          f'train_loss: {train_loss:.2f} | '
                          f'valid_loss: {val_loss:.3f}')
            not_learn_params = torch.cat((not_learn_params, global_model.state_dict()['hyp']), 0)
            torch.save(not_learn_params, f'checkpoints/{args.case}/{args.exp_name}/{args.model}/params_P={args.P}_batch={not_learn_params.shape[0]}_prob={args.graph_prob}_size={ll + args.max_iter_seg}.pt')
    torch.save(not_learn_params, f'checkpoints/{args.case}/{args.exp_name}/{args.model}/params_P={args.P}_batch={args.max_iter}_prob={args.graph_prob}.pt')

    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.0f} min')
