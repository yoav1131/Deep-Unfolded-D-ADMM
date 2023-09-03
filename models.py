import torch
import torch.nn as nn
import gc

class DADMMLRDiff(nn.Module):
    def __init__(self, hyp, ll, no_hyp):
        super(DADMMLRDiff, self).__init__()
        if ll == 0:
            self.hyp = nn.Parameter(hyp)
            self.ll = ll
            self.no_hyp = None
        else:
            self.hyp = nn.Parameter(hyp)
            self.ll = ll
            self.no_hyp = no_hyp


    def forward(self, input_vars, neighbors, color_partition, args, labels, test):
        """
        Inputs:
         # The data inputs (observations)
         # P - No. of nodes
         Network variables:
         - vars_network is a class containing information about the network (graph):
         # neighbors - array with P agents, where each entry j contains the neighbors of agent p
         # color partition - array of the number of (proper) colors of the network graph. Each
           entry contains a vector with the nodes that have that color.
           For example, for a network with 6 nodes,
           {[1,2], [3,4,5], 6} would mean that we have three colors:
           nodes 1 and 2 get the first color, 3, 4, and 5 the second,
           and node 6 gets the third color.
         - MAX_ITER - number of D-ADMM iterations
        """

        #--- initializing variables
        m = input_vars.inputs.shape[0]
        m_p = args.batch_size
        mu = torch.zeros((args.P, args.batch_size, 28 * 28, 1), dtype=torch.float64).to(args.device)  # Dual variable of a
        lamda = torch.zeros((args.P, args.batch_size, 1, 1), dtype=torch.float64).to(args.device)  # Dual variable of omega
        a = torch.zeros((args.P, args.batch_size, 28 * 28, 1), dtype=torch.float64)
        omega = torch.zeros((args.P, args.batch_size, 1, 1), dtype=torch.float64)
        torch.manual_seed(a.flatten().shape[0])
        a = torch.nn.init.normal_(a).to(args.device)
        omega = torch.nn.init.uniform_(omega).to(args.device)
        n_a = a.shape[2]
        n_omega = omega.shape[2]
        num_colors = len(color_partition)
        loss_arr, acc_arr = [], []
        weights_arr, biases_arr = [], []
        if test:
            MAX_ITER = args.max_iter
        else:
            MAX_ITER = args.max_iter_seg

        """DADMM algorithm"""
        for k in range(MAX_ITER + self.ll):
            if test and k == MAX_ITER:
                break
            if k + 1 == MAX_ITER:
                args.batch_size = int(m_p) + m % args.P
            for color in range(num_colors):
                a_aux = a.clone()
                omega_aux = omega.clone()
                for p in color_partition[color]:
                    neighbs = neighbors[p]
                    Dp = len(neighbs)
                    # --- Determine the sum of the X's of the neighbors ---
                    sum_neighbs_a = torch.zeros(args.batch_size, n_a, 1).to(args.device)
                    sum_neighbs_omega = torch.zeros(args.batch_size, n_omega, 1).to(args.device)
                    for j in range(Dp):
                        sum_neighbs_a = sum_neighbs_a + a[neighbs[j]]
                        sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]]
                    # primal update
                    a_aux[p] = self.grad_alpha(p, k, a[p], omega[p], mu[p], labels[p], sum_neighbs_a, Dp, input_vars.inputs[p])
                    omega_aux[p] = self.grad_omega(p, k, a[p], omega[p], lamda[p], labels[p], sum_neighbs_omega, Dp, input_vars.inputs[p])
                    if float(torch.max(torch.abs(omega_aux[p][:, 0, 0]))) > 100:
                        print(torch.max(torch.abs(omega_aux[p][:, 0, 0])))
                    gc.collect()
                a = a_aux.clone()
                omega = omega_aux.clone()
                gc.collect()

            # Output
            neighbs = neighbors[0]
            Dp = len(neighbs)
            # --- Determine the sum of the X's of the neighbors ---
            sum_neighbs_a = torch.zeros(args.batch_size, n_a, 1).to(args.device)
            sum_neighbs_omega = torch.zeros(args.batch_size, n_omega, 1).to(args.device)
            for j in range(Dp):
                sum_neighbs_a = sum_neighbs_a + a[neighbs[j]]
                sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]]
            # dual update
            if k >= self.ll:
                mu_ = [mu[0].clone().detach() + (torch.abs(self.hyp[k - self.ll][0][3])) * (Dp * a[0] - sum_neighbs_a)]
                lamda_ = [lamda[0].clone().detach() + (torch.abs(self.hyp[k - self.ll][0][4])) * (Dp * omega[0] - sum_neighbs_omega)]
            else:
                mu_ = [mu[0].clone().detach() + (torch.abs(self.no_hyp[k][0][3])) * (Dp * a[0] - sum_neighbs_a)]
                lamda_ = [lamda[0].clone().detach() + (torch.abs(self.no_hyp[k][0][4])) * (Dp * omega[0] - sum_neighbs_omega)]
            for pp in range(1, args.P):
                neighbs = neighbors[pp]
                Dp = len(neighbs)
                # --- Determine the sum of the X's of the neighbors ---
                sum_neighbs_a = torch.zeros(args.batch_size, n_a, 1).to(args.device)
                sum_neighbs_omega = torch.zeros(args.batch_size, n_omega, 1).to(args.device)
                for j in range(Dp):
                    sum_neighbs_a = sum_neighbs_a + a[neighbs[j]]
                    sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]]
                if k >= self.ll:
                    mu_.append(mu[pp].clone().detach() + (torch.abs(self.hyp[k - self.ll][pp][3])) * (Dp * a[pp] - sum_neighbs_a))
                    lamda_.append(lamda[pp].clone().detach() + (torch.abs(self.hyp[k - self.ll][pp][4])) * (Dp * omega[pp] - sum_neighbs_omega))
                else:
                    mu_.append(mu[pp].clone().detach() + (torch.abs(self.no_hyp[k][pp][3])) * (Dp * a[pp] - sum_neighbs_a))
                    lamda_.append(lamda[pp].clone().detach() + (torch.abs(self.no_hyp[k][pp][4])) * (Dp * omega[pp] - sum_neighbs_omega))
                gc.collect()
            mu = torch.stack(mu_)
            lamda = torch.stack(lamda_)
            gc.collect()
            if test:
                weights, biases, loss_arr, acc_arr = self.test_set(a, omega, k, input_vars.inputs, labels, weights_arr, biases_arr, loss_arr, acc_arr)
        if not test:
            return a, omega, loss_arr, acc_arr
        else:
            return weights, biases, loss_arr, acc_arr

    def grad_alpha(self, p, k, a, omega, mu, labels, sum_neighbs, Dp, inputs):
        if k >= self.ll:
            return a - (torch.abs(self.hyp[k - self.ll][p][1])) * (inputs @ torch.transpose(inputs, 1, 2) @ a
                                                                                 + inputs @ omega - inputs @ labels +
                                                                                torch.abs(self.hyp[k - self.ll][p][0]) * a * Dp +
                                                                      Dp * mu - torch.abs(self.hyp[k - self.ll][p][0]) * sum_neighbs)
        else:
            return a - (torch.abs(self.no_hyp[k][p][1])) * (inputs @ torch.transpose(inputs, 1, 2) @ a
                                                                                 + inputs @ omega - inputs @ labels +
                                                                                torch.abs(self.no_hyp[k][p][0]) * a * Dp +
                                                                      Dp * mu - torch.abs(self.no_hyp[k][p][0]) * sum_neighbs)

    def grad_omega(self, p, k, a, omega, lamda, labels, sum_neighbs, Dp, inputs):
        if k >= self.ll:
            return omega - (torch.abs(self.hyp[k - self.ll][p][-1])) * ((torch.transpose(a, 1, 2) @ inputs)
                                                                                 + omega - labels +
                                                                (torch.abs(self.hyp[k - self.ll][p][2])) * omega * Dp + lamda * Dp
                                                                          - (torch.abs(self.hyp[k - self.ll][p][2])) * sum_neighbs)
        else:
            return omega - (torch.abs(self.no_hyp[k][p][-1])) * ((torch.transpose(a, 1, 2) @ inputs)
                                                              + omega - labels +
                                                              (torch.abs(self.no_hyp[k][p][2])) * omega * Dp + lamda * Dp
                                                              - (torch.abs(self.no_hyp[k][p][2])) * sum_neighbs)

    def test_set(self, a, omega, k, inputs, labels, weights, biases, loss_arr, acc_arr):
        weights.append(a)
        biases.append(omega)
        loss_, acc_ = 0, 0
        mse_loss = nn.MSELoss()
        for jj in range(a.shape[0]):
            y_hat = (torch.transpose(a[jj], 1, 2) @ inputs[jj] + omega[jj])[:, :, 0]
            for ii in range(y_hat.shape[0]):
                # calculate the loss
                loss_ += mse_loss(torch.abs(y_hat[ii].flatten()), labels[jj][ii].flatten())
                # calculate the accuracy
                if torch.abs(y_hat[ii]) % 1 > 0.8 or torch.abs(y_hat[ii]) % 1 < 0.2:
                    acc_ += (torch.round(torch.abs(y_hat[ii].flatten())) == labels[jj][ii].flatten()).sum()
        loss = loss_ / (a.shape[0] * y_hat.shape[0])
        acc = (acc_ / (a.shape[0] * y_hat.shape[0])) * 100
        # print(f'Iter: {k}; Loss: {loss}, accuracy: {acc}')
        loss_arr.append(loss.item())
        acc_arr.append(acc.item())
        return weights, biases, loss_arr, acc_arr



class DADMMLRSame(nn.Module):
    def __init__(self, hyp, ll, no_hyp):
        super(DADMMLRSame, self).__init__()
        if ll == 0:
            self.hyp = nn.Parameter(hyp)
            self.ll = ll
            self.no_hyp = None
        else:
            self.hyp = nn.Parameter(hyp)
            self.ll = ll
            self.no_hyp = no_hyp


    def forward(self, input_vars, neighbors, color_partition, args, labels, test):
        """
        Inputs:
         # The data inputs (observations)
         # P - No. of nodes
         Network variables:
         - vars_network is a class containing information about the network (graph):
         # neighbors - array with P agents, where each entry j contains the neighbors of agent p
         # color partition - array of the number of (proper) colors of the network graph. Each
           entry contains a vector with the nodes that have that color.
           For example, for a network with 6 nodes,
           {[1,2], [3,4,5], 6} would mean that we have three colors:
           nodes 1 and 2 get the first color, 3, 4, and 5 the second,
           and node 6 gets the third color.
         - MAX_ITER - number of D-ADMM iterations
        """

        # initializing variables
        m = input_vars.inputs.shape[0]
        m_p = args.batch_size
        mu = torch.zeros((args.P, args.batch_size, 28*28, 1), dtype=torch.float64).to(args.device)  # Dual variable of a
        lamda = torch.zeros((args.P, args.batch_size, 1, 1), dtype=torch.float64).to(args.device)  # Dual variable of omega
        a = torch.zeros((args.P, args.batch_size, 28 * 28, 1), dtype=torch.float64)
        omega = torch.zeros((args.P, args.batch_size, 1, 1), dtype=torch.float64)
        torch.manual_seed(a.flatten().shape[0])
        a = torch.nn.init.normal_(a).to(args.device)
        omega = torch.nn.init.uniform_(omega).to(args.device)
        n_a = a.shape[2]
        n_omega = omega.shape[2]
        num_colors = len(color_partition)
        loss_arr, acc_arr = [], []
        weights, biases = [], []
        if test:
            MAX_ITER = args.max_iter
        else:
            MAX_ITER = args.max_iter_seg

        """DADMM algorithm"""
        for k in range(MAX_ITER + self.ll):
            if test and k == MAX_ITER:
                break
            if k + 1 == MAX_ITER:
                args.batch_size = int(m_p) + m % args.P
            for color in range(num_colors):
                a_aux = a.clone()
                omega_aux = omega.clone()
                for p in color_partition[color]:
                    neighbs = neighbors[p]
                    Dp = len(neighbs)

                    # --- Determine the sum of the X's of the neighbors ---
                    sum_neighbs_a = torch.zeros(args.batch_size, n_a, 1).to(args.device)
                    sum_neighbs_omega = torch.zeros(args.batch_size, n_omega, 1).to(args.device)
                    for j in range(Dp):
                        sum_neighbs_a = sum_neighbs_a + a[neighbs[j]]
                        sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]]
                    # primal update
                    a_aux[p] = self.grad_alpha(k, a[p], omega[p], mu[p], labels[p], sum_neighbs_a, Dp, input_vars.inputs[p])
                    omega_aux[p] = self.grad_omega(k, a[p], omega[p], lamda[p], labels[p], sum_neighbs_omega, Dp, input_vars.inputs[p])
                    if float(torch.max(torch.abs(omega_aux[p][:, 0, 0]))) > 100:
                        print(torch.max(torch.abs(omega_aux[p][:, 0, 0])))
                    gc.collect()
                a = a_aux.clone()
                omega = omega_aux.clone()
                gc.collect()

            # Output
            neighbs = neighbors[0]
            Dp = len(neighbs)
            # --- Determine the sum of the X's of the neighbors ---
            sum_neighbs_a = torch.zeros(args.batch_size, n_a, 1).to(args.device)
            sum_neighbs_omega = torch.zeros(args.batch_size, n_omega, 1).to(args.device)
            for j in range(Dp):
                sum_neighbs_a = sum_neighbs_a + a[neighbs[j]]
                sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]]
            # dual update
            if k >= self.ll:
                mu_ = [mu[0].clone().detach() + (torch.abs(self.hyp[k - self.ll][3])) * (Dp * a[0] - sum_neighbs_a)]
                lamda_ = [lamda[0].clone().detach() + (torch.abs(self.hyp[k - self.ll][4])) * (Dp * omega[0] - sum_neighbs_omega)]
            else:
                mu_ = [mu[0].clone().detach() + (torch.abs(self.no_hyp[k][3])) * (Dp * a[0] - sum_neighbs_a)]
                lamda_ = [lamda[0].clone().detach() + (torch.abs(self.no_hyp[k][4])) * (Dp * omega[0] - sum_neighbs_omega)]
            for pp in range(1, args.P):
                neighbs = neighbors[pp]
                Dp = len(neighbs)
                # --- Determine the sum of the X's of the neighbors ---
                sum_neighbs_a = torch.zeros(args.batch_size, n_a, 1).to(args.device)
                sum_neighbs_omega = torch.zeros(args.batch_size, n_omega, 1).to(args.device)
                for j in range(Dp):
                    sum_neighbs_a = sum_neighbs_a + a[neighbs[j]]
                    sum_neighbs_omega = sum_neighbs_omega + omega[neighbs[j]]
                if k >= self.ll:
                    mu_.append(mu[pp].clone().detach() + (torch.abs(self.hyp[k - self.ll][3])) * (Dp * a[pp] - sum_neighbs_a))
                    lamda_.append(lamda[pp].clone().detach() + (torch.abs(self.hyp[k - self.ll][4])) * (Dp * omega[pp] - sum_neighbs_omega))
                else:
                    mu_.append(mu[pp].clone().detach() + (torch.abs(self.no_hyp[k][3])) * (Dp * a[pp] - sum_neighbs_a))
                    lamda_.append(lamda[pp].clone().detach() + (torch.abs(self.no_hyp[k][4])) * (Dp * omega[pp] - sum_neighbs_omega))
                gc.collect()
            mu = torch.stack(mu_)
            lamda = torch.stack(lamda_)
            gc.collect()
            if test:
                weights.append(a)
                biases.append(omega)
                loss_, acc_ = 0, 0
                mse_loss = nn.MSELoss()
                for jj in range(a.shape[0]):
                    y_hat = (torch.transpose(a[jj], 1, 2) @ input_vars.inputs[jj] + omega[jj])[:, :, 0]
                    for ii in range(y_hat.shape[0]):
                        # calculate the loss
                        loss_ += mse_loss(torch.abs(y_hat[ii].flatten()), labels[jj][ii].flatten())
                        # calculate the accuracy
                        if torch.abs(y_hat[ii]) % 1 > 0.8 or torch.abs(y_hat[ii]) % 1 < 0.2:
                            acc_ += (torch.round(torch.abs(y_hat[ii].flatten())) == labels[jj][ii].flatten()).sum()
                loss = loss_ / (a.shape[0] * y_hat.shape[0])
                acc = (acc_ / (a.shape[0] * y_hat.shape[0])) * 100
                # print(f'Iter: {k}; Loss: {loss}, accuracy: {acc}')
                loss_arr.append(loss.item())
                acc_arr.append(acc.item())
        if not test:
            return a, omega, loss_arr, acc_arr
        else:
            return weights, biases, loss_arr, acc_arr

    def grad_alpha(self, k, a, omega, mu, labels, sum_neighbs, Dp, inputs):
        if k >= self.ll:
            return a - (torch.abs(self.hyp[k - self.ll][1])) * (inputs @ torch.transpose(inputs, 1, 2) @ a
                                                                                 + inputs @ omega - inputs @ labels +
                                                                                torch.abs(self.hyp[k - self.ll][0]) * a * Dp +
                                                                      Dp * mu - torch.abs(self.hyp[k - self.ll][0]) * sum_neighbs)
        else:
            return a - (torch.abs(self.no_hyp[k][1])) * (inputs @ torch.transpose(inputs, 1, 2) @ a
                                                                                 + inputs @ omega - inputs @ labels +
                                                                                torch.abs(self.no_hyp[k][0]) * a * Dp +
                                                                      Dp * mu - torch.abs(self.no_hyp[k][0]) * sum_neighbs)

    def grad_omega(self, k, a, omega, lamda, labels, sum_neighbs, Dp, inputs):
        if k >= self.ll:
            return omega - (torch.abs(self.hyp[k - self.ll][-1])) * ((torch.transpose(a, 1, 2) @ inputs)
                                                                                 + omega - labels +
                                                                (torch.abs(self.hyp[k - self.ll][2])) * omega * Dp + lamda * Dp
                                                                          - (torch.abs(self.hyp[k - self.ll][2])) * sum_neighbs)
        else:
            return omega - (torch.abs(self.no_hyp[k][-1])) * ((torch.transpose(a, 1, 2) @ inputs)
                                                              + omega - labels +
                                                              (torch.abs(self.no_hyp[k][2])) * omega * Dp + lamda * Dp
                                                              - (torch.abs(self.no_hyp[k][2])) * sum_neighbs)

class DADMMLASSO(nn.Module):
    def __init__(self, hyp, ll, args, no_hyp):
        super(DADMMLASSO, self).__init__()
        self.args = args
        if ll == 0:
            self.hyp = nn.Parameter(hyp)
            self.ll = ll
            self.no_hyp = None
        else:
            self.hyp = nn.Parameter(hyp)
            self.ll = ll
            self.no_hyp = no_hyp

    def forward(self, vars_prob, neighbors, color_partition, args, labels, test):
        """
        Inputs:
         # P - No. of nodes
         # The sensing matrix A (500x2000)
         # The data inputs (observations)
         # m_p - splitting factor according to nodes number
         Network variables:
         - The network variables containing information about the network (graph):
         # neighbors - array with P agents, where each entry j contains the neighbors of agent p
         # color partition - array of the number of (proper) colors of the network graph. Each
           entry contains a vector with the nodes that have that color.
           For example, for a network with 6 nodes,
           {[1,2], [3,4,5], 6} would mean that we have three colors:
           nodes 1 and 2 get the first color, 3, 4, and 5 the second,
           and node 6 gets the third color.
         - MAX_ITER - number of D-ADMM iterations

        """
        # --- initializing variables
        m, n = vars_prob.A_BPDN.shape
        m_p = m / args.P  # Number of rows of A that each node stores
        vars_prob.m_p = m_p
        X = torch.zeros((args.P, args.batch_size, 2000, 1), dtype=torch.float64)
        U = torch.zeros((args.P, args.batch_size, 2000, 1), dtype=torch.float64)
        size = X.shape[1]
        n = X.shape[2]
        num_colors = len(color_partition)
        loss_arr = []

        """DADMM algorithm"""
        for k in range(args.max_iter_seg + self.ll):
            if test and k == args.max_iter_seg:
                break
            for color in range(num_colors):
                X_aux = X.clone()
                for p in color_partition[color]:
                    neighbs = neighbors[p]
                    Dp = len(neighbs)
                    # --- Determine the sum of the X's of the neighbors ---
                    sum_neighbs = torch.zeros(size, n, 1)
                    for j in range(Dp):
                        sum_neighbs = sum_neighbs + X[neighbs[j]].clone().detach()
                    X_aux[p] = self.bpdn_rp_solver(p, U[p], sum_neighbs, Dp, X[p], vars_prob, k)
                X = X_aux.clone()

            # Output
            neighbs = neighbors[0]
            Dp = len(neighbs)
            # --- Determine the sum of the X's of the neighbors ---
            sum_neighbs = torch.zeros(size, n, 1)
            for j in range(Dp):
                sum_neighbs = sum_neighbs + X[neighbs[j]]
            if k >= self.ll:
                U_ = [U[0].clone().detach() + (torch.abs(self.hyp[k - self.ll][0][-1])) * (Dp * X[0] - sum_neighbs)]
            else:
                U_ = [U[0].clone().detach() + (torch.abs(self.no_hyp[k][0][-1])) * (Dp * X[0] - sum_neighbs)]
            for pp in range(1, args.P):
                neighbs = neighbors[pp]
                Dp = len(neighbs)
                # --- Determine the sum of the X's of the neighbors ---
                sum_neighbs = torch.zeros(args.batch_size, n, 1)
                for j in range(Dp):
                    sum_neighbs = sum_neighbs + X[neighbs[j]]
                if k >= self.ll:
                    U_.append(U[pp].clone().detach() + (torch.abs(self.hyp[k - self.ll][pp][-1])) * (
                            Dp * X[pp] - sum_neighbs))
                else:
                    U_.append(U[pp].clone().detach() + (torch.abs(self.no_hyp[k][pp][-1])) * (Dp * X[pp] - sum_neighbs))
            U = torch.stack(U_)
            if test:
                loss_ = 0
                mse_loss = nn.MSELoss()
                for ii in range(X.shape[0]):
                    for jj in range(labels.shape[0]):
                        loss_ += mse_loss(X[ii][jj].flatten(), labels[jj].flatten())
                loss = loss_ / (X.shape[0] * labels.shape[0])
                loss_arr.append(loss.item())
        return X, U, loss_arr

    def bpdn_rp_solver(self, p, U, sum_neighbs, Dp, X, vars_prob, k):
        A_full = vars_prob.A_BPDN
        b_full = vars_prob.inputs
        m_p = vars_prob.m_p
        Ap = A_full[int(p * m_p): int((p + 1) * m_p)]
        bp = b_full[:, int(p * m_p): int((p + 1) * m_p)]
        if k >= self.ll:
            return X.clone().detach() - (torch.abs(self.hyp[k - self.ll][p][1])) * (Ap.T @ Ap @ X - Ap.T @ bp +
                                                        Dp * (torch.abs(self.hyp[k - self.ll][p][0])) * X +
                                                        (torch.abs(self.hyp[k - self.ll][p][2])) * torch.sign(X) + Dp * U -
                                                        (torch.abs(self.hyp[k - self.ll][p][0])) * sum_neighbs)
        else:
            return X.clone().detach() - (torch.abs(self.no_hyp[k][p][1])) * (
                        Ap.T @ Ap @ X - Ap.T @ bp + Dp * (torch.abs(self.no_hyp[k][p][0])) * X +
                        (torch.abs(self.no_hyp[k][p][2])) * torch.sign(X) + Dp * U -
                        (torch.abs(self.no_hyp[k][p][0])) * sum_neighbs)


