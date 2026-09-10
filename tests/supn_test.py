from cholespy import CholeskySolverF, SupnCholeskySolverF, MatrixType, inspect
import torch
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()


from supn_utils import get_sparse_LT_matrix_index_values, get_num_off_diag_weights, MIN_DIAG_VALUE

# device = 'cpu'
device = 'cuda'

width = 5
height = 4
local_connection_dist = 1

np.random.seed(42)

log_diag_weights = torch.tensor(0.1 * np.random.randn(1, 1, width, height).astype(float),
                                dtype=torch.float, device=device)
off_diag_weights = torch.tensor(0.1 * np.random.randn(1, get_num_off_diag_weights(local_connection_dist),
                                                      width, height).astype(float),
                                dtype=torch.float, device=device)


class SUPNSolver:
    def __init__(self, log_diag_weights, off_diag_weights, local_connection_dist):
        rows, cols, index_values, shape = get_sparse_LT_matrix_index_values(log_diag_weights,
                                                                            off_diag_weights,
                                                                            local_connection_dist,
                                                                            use_transpose=True)
        self._device = log_diag_weights.device
        assert log_diag_weights.device == off_diag_weights.device

        raw_values = self._get_raw_values(log_diag_weights, off_diag_weights)

        assert raw_values.ndim == 4
        assert raw_values.shape[0] == 1

        raw_values = raw_values.flatten()

        sparse_LT_coo = torch.sparse_coo_tensor(indices=torch.stack([rows, cols]),
                                                values=index_values,
                                                size=shape,
                                                dtype=torch.float)

        assert shape[1] == shape[0]

        sparse_LT_csr = sparse_LT_coo.to_sparse_csr()

        upper_csr_rows = sparse_LT_csr.crow_indices().int()
        upper_csr_cols = sparse_LT_csr.col_indices().int()
        upper_csr_index_data = sparse_LT_csr.values()

        sparse_L_csr = torch.transpose(sparse_LT_coo, 0, 1).to_sparse_csr()

        lower_csr_rows = sparse_L_csr.crow_indices().int()
        lower_csr_cols = sparse_L_csr.col_indices().int()
        lower_csr_index_data = sparse_L_csr.values()

        csr_n_rows = sparse_L_csr.shape[0]

        # Remember to take 1 off to go back to zero based index!!
        upper_csr_indices = upper_csr_index_data.int() - 1
        lower_csr_indices = lower_csr_index_data.int() - 1

        self._supn_chol_solver = SupnCholeskySolverF(csr_n_rows,
                                          lower_csr_rows, lower_csr_cols, lower_csr_indices,
                                          upper_csr_rows, upper_csr_cols, upper_csr_indices,
                                          raw_values)

    def _get_raw_values(self, log_diag_weights, off_diag_weights):
        return torch.concat([torch.exp(log_diag_weights) + MIN_DIAG_VALUE,
                                   off_diag_weights], dim=1)

    def test_sparse_matrices(self, log_diag_weights, off_diag_weights):
        raw_values = self._get_raw_values(log_diag_weights, off_diag_weights)
        raw_values = raw_values.flatten()

        def get_matrix(lower):
            rr = torch.ones(self._supn_chol_solver.get_n_rows() + 1, dtype=torch.int32, device=self._device)
            cc = torch.ones(self._supn_chol_solver.get_n_entries(), dtype=torch.int32, device=self._device)
            dd = torch.ones(self._supn_chol_solver.get_n_entries(), dtype=torch.int32, device=self._device)
            vv = torch.ones(self._supn_chol_solver.get_n_raw_data(), device=self._device)

            self._supn_chol_solver.debug_print(rr, cc, dd, vv, lower)

            return torch.sparse_csr_tensor(rr, cc, raw_values[dd.long()], size=[height * width, height * width], dtype=torch.float)

        LL = get_matrix(lower=True)
        print('LL', LL, '\n', LL.to_dense())

        LT = get_matrix(lower=False)
        print('LT', LT, '\n', LT.to_dense())

        return LL, LT

    def solve_with_upper(self, log_diag_weights, off_diag_weights, rhs):
        return self._solve(log_diag_weights, off_diag_weights, rhs, skip_lower=True)

    def solve_with_precision(self, log_diag_weights, off_diag_weights, rhs):
        return self._solve(log_diag_weights, off_diag_weights, rhs, skip_lower=False)

    def _solve(self, log_diag_weights, off_diag_weights, rhs, skip_lower):
        assert rhs.ndim == 4
        assert rhs.shape[0] == 1
        # assert rhs.shape[1] == 1
        assert rhs.shape[2] * rhs.shape[3] == self._supn_chol_solver.get_n_rows()

        rhs_shape = rhs.shape
        if rhs_shape[1] > 1:
            rhs = torch.transpose(rhs.view(-1, width * height), 0, 1)
            rhs = rhs.view(width * height, -1)
            rhs = rhs.contiguous()
            print('rhs', rhs.shape)
        else:
            rhs = rhs.view(self._supn_chol_solver.get_n_rows())

        raw_values = self._get_raw_values(log_diag_weights, off_diag_weights)
        raw_values = raw_values.flatten()

        assert raw_values.shape[0] == self._supn_chol_solver.get_n_raw_data()

        solution = torch.zeros_like(rhs)

        self._supn_chol_solver.solve(raw_values, rhs, solution, skip_lower)

        if rhs_shape[1] > 1:
            solution = torch.transpose(solution, 0, 1)

        solution = solution.view(*rhs_shape)

        return solution


supn_solver = SUPNSolver(log_diag_weights, off_diag_weights, local_connection_dist)

LL, LT = supn_solver.test_sparse_matrices(log_diag_weights, off_diag_weights)

# Test single RHS precision solve..
rhs = torch.ones((1, 1, width, height), dtype=torch.float, device=device)
x = supn_solver.solve_with_precision(log_diag_weights, off_diag_weights, rhs)
print('x', x)
print('LL @ LT @ x', LL.to_sparse_coo() @ LT.to_sparse_coo() @ x.view(width * height))

# Test multiple RHS precision solve..
rhs = torch.concat([1.0 * torch.ones((1, 1, width, height), dtype=torch.float, device=device),
                    0.5 * torch.ones((1, 1, width, height), dtype=torch.float, device=device)], dim=1)
x = supn_solver.solve_with_precision(log_diag_weights, off_diag_weights, rhs)
print('x', x)
print('LL @ LT @ x', LL.to_sparse_coo() @ LT.to_sparse_coo() @ x.view(-1, width * height).t())

# Test multiple RHS just upper LT solve..
x = supn_solver.solve_with_upper(log_diag_weights, off_diag_weights, rhs)
print('x', x)
print('LT @ x', LT.to_sparse_coo() @ x.view(-1, width * height).t())
