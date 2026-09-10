import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F

# import torch_sparse_solve as tss
# import scipy.sparse as sparse

MIN_DIAG_VALUE = 1e-6


def build_off_diag_filters(local_connection_dist, use_transpose=True, device=None, dtype=torch.float):
    """Create the conv2d filter weights for the off-diagonal components of the sparse chol.

    NOTE: Important to specify device if things might run under cuda since constants are created and need to be
        on the correct device.

    Parameters:
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True - usually what we want for the jacobi sampling.
        device: Specify the device to create the constants on (i.e. cpu vs gpu).

    Returns:
        tri_off_diag_filters(tensor): [num_off_diag_weights x 1 x filter_size x filter_size] Conv2d kernel filters.
    """
    filter_size = 2 * local_connection_dist + 1
    filter_size_sq = filter_size * filter_size
    filter_size_sq_2 = filter_size_sq // 2

    if use_transpose:
        tri_off_diag_filters = torch.cat((torch.zeros(filter_size_sq_2, (filter_size_sq_2 + 1),
                                                      device=device, dtype=dtype),
                                          torch.eye(filter_size_sq_2,
                                                    device=device, dtype=dtype)), dim=1)
    else:
        tri_off_diag_filters = torch.cat((torch.fliplr(torch.eye(filter_size_sq_2,
                                                                 device=device, dtype=dtype)),
                                          torch.zeros(filter_size_sq_2, (filter_size_sq_2 + 1),
                                                      device=device, dtype=dtype)), dim=1)

    tri_off_diag_filters = torch.reshape(tri_off_diag_filters, (filter_size_sq_2, 1, filter_size, filter_size))

    return tri_off_diag_filters


def get_num_off_diag_weights(local_connection_dist):
    filter_size = 2 * local_connection_dist + 1
    filter_size_sq = filter_size * filter_size
    filter_size_sq_2 = filter_size_sq // 2
    return filter_size_sq_2


def get_sparse_LT_matrix_index_values(log_diag_weights, off_diag_weights, local_connection_dist,
                                      use_transpose=True):
    """
    Parameters:
        log_diag_weights(tensor): [BATCH x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [BATCH x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).

    :param log_diag_weights:
    :param off_diag_weights:
    :param local_connection_dist:
    :param use_transpose:
    :return:
    """
    assert log_diag_weights.ndim == 4
    assert off_diag_weights.ndim == 4

    # We can only do batch size 1 at the moment..
    assert log_diag_weights.shape[0] == 1
    assert off_diag_weights.shape[0] == 1

    assert log_diag_weights.device == off_diag_weights.device
    device = off_diag_weights.device
    dtype = off_diag_weights.dtype

    num_off_diag_weights = get_num_off_diag_weights(local_connection_dist=local_connection_dist)
    tri_off_diag_filters = build_off_diag_filters(local_connection_dist=local_connection_dist,
                                                  use_transpose=use_transpose,
                                                  device=device,
                                                  dtype=dtype)

    batch_size = log_diag_weights.shape[0]
    assert log_diag_weights.shape[1] == 1
    im_size_H = log_diag_weights.shape[2]
    im_size_W = log_diag_weights.shape[3]

    assert batch_size == 1

    diag_values = torch.exp(log_diag_weights) + MIN_DIAG_VALUE

    diag_values = torch.reshape((torch.arange(diag_values.numel(), dtype=dtype, device=device) + 1.0), diag_values.shape)
    off_diag_weights = torch.reshape(torch.arange(off_diag_weights.numel(), dtype=dtype, device=device) +
                                     diag_values.numel() + 1.0, off_diag_weights.shape)

    index_input = torch.arange(im_size_H * im_size_W, device=device).view(1, 1, im_size_H, im_size_W) + 1

    off_diag_indices = F.conv2d(index_input.view(-1, 1, im_size_H, im_size_W).float(), tri_off_diag_filters,
                                padding=local_connection_dist, stride=1)

    all_indices_col = torch.cat((index_input, off_diag_indices), dim=-3)

    all_indices_row = torch.cat((1 + num_off_diag_weights) * [index_input], dim=-3)

    all_values = torch.cat((diag_values, off_diag_weights), dim=-3)

    all_indices_col = all_indices_col.flatten().long()
    all_indices_row = all_indices_row.flatten().long()
    all_values = all_values.flatten()

    all_indices_col_used = all_indices_col[all_indices_col > 0]
    all_indices_row_used = all_indices_row[all_indices_col > 0]
    all_values_used = all_values[all_indices_col > 0]

    all_indices_col_used -= 1
    all_indices_row_used -= 1

    rows = all_indices_row_used
    cols = all_indices_col_used
    values = all_values_used
    shape = [im_size_H * im_size_W, im_size_H * im_size_W]

    return rows, cols, values, shape

    # sparse_LT_coo = torch.sparse_coo_tensor(indices=torch.stack([all_indices_row_used, all_indices_col_used]),
    #                                         values=all_values_used,
    #                                         size=[im_size_H * im_size_W, im_size_H * im_size_W],
    #                                         dtype=torch.double)
    #
    # return sparse_LT_coo