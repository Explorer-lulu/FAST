import csv
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import expm

from .metrics import mean_absolute_error, mean_squared_error, masked_mape_np


def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                               label_start_idx, num_for_predict,
                               7 * 24, points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days,
                              label_start_idx, num_for_predict,
                              24, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                               label_start_idx, num_for_predict,
                               1, points_per_hour)
    if not hour_indices:
        return None

    week_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in week_indices], axis=0)
    day_sample = np.concatenate([data_sequence[i: j]
                                 for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def get_adjacency_matrix(distance_df_filename, num_of_vertices):

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    return A


def get_adjacency_to_diffision_matrix(distance_df_filename, num_of_vertices, kt):
    '''
    Parameters
    // Modify by GDC data ppr method, give staic paramter alpha as 0.1
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    A: np.ndarray, orignal adjacency matrix

    Returns
    ----------
    
    DM : np.ndarray, Diffision Matrix by GDC

    
    Now: alpha == 0.08 
    '''
    alpha: float = 0.08

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1
    
    A_tmatrix = A + np.eye(num_of_vertices)
    
    D_tmatrix = np.diag(1/np.sqrt(A_tmatrix.sum(axis=1)))

    H_dot = D_tmatrix @ A_tmatrix @ D_tmatrix

    PPR_result = alpha * np.linalg.inv(np.eye(num_of_vertices) - (1 - alpha) * H_dot)

    row_index = np.arange(num_of_vertices)

    PPR_result[PPR_result.argsort(axis=0)[:num_of_vertices - kt], row_index] = 0.

    norm = PPR_result.sum(axis=0)

    norm[norm <= 0] =1

    DM = PPR_result/norm

    return DM



def scaled_Laplacian(W):

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def compute_val_loss(net, val_loader, loss_function, sw, epoch):
    val_loader_length = len(val_loader)
    tmp = []
    for index, (val_w, val_d, val_r, val_t) in enumerate(val_loader):
        output = net([val_w, val_d, val_r])
        l = loss_function(output, val_t)
        tmp.extend(l.asnumpy().tolist())
        print('validation batch %s / %s, loss: %.2f' % (
            index + 1, val_loader_length, l.mean().asscalar()))

    validation_loss = sum(tmp) / len(tmp)
    sw.add_scalar(tag='validation_loss',
                  value=validation_loss,
                  global_step=epoch)
    print('epoch: %s, validation loss: %.2f' % (epoch, validation_loss))


def predict(net, test_loader):
    test_loader_length = len(test_loader)
    prediction = []
    for index, (test_w, test_d, test_r, _) in enumerate(test_loader):
        prediction.append(net([test_w, test_d, test_r]).asnumpy())
        print('predicting testing set batch %s / %s' % (index + 1,
                                                        test_loader_length))
    prediction = np.concatenate(prediction, 0)
    return prediction


def evaluate(net, test_loader, true_value, num_of_vertices, sw, epoch):

    prediction = predict(net, test_loader)
    prediction = (prediction.transpose((0, 2, 1))
                  .reshape(prediction.shape[0], -1))
    for i in [3, 6, 12]:
        print('current epoch: %s, predict %s points' % (epoch, i))

        mae = mean_absolute_error(true_value[:, : i * num_of_vertices],
                                  prediction[:, : i * num_of_vertices])
        rmse = mean_squared_error(true_value[:, : i * num_of_vertices],
                                  prediction[:, : i * num_of_vertices]) ** 0.5
        mape = masked_mape_np(true_value[:, : i * num_of_vertices],
                              prediction[:, : i * num_of_vertices], 0)

        print('MAE: %.2f' % (mae))
        print('RMSE: %.2f' % (rmse))
        print('MAPE: %.2f' % (mape))
        print()

def evaluate_best(net, test_loader, true_value, num_of_vertices, epoch):
    prediction = predict(net, test_loader)
    prediction = (prediction.transpose((0, 2, 1))
                  .reshape(prediction.shape[0], -1))
    i = 1
    print('current epoch: %s, predict %s points' % (epoch, i))
    mae = mean_absolute_error(true_value[:, : i * num_of_vertices],
                                  prediction[:, : i * num_of_vertices])
    print('MAE: %.2f' % (mae))
    #print('RMSE: %.2f' % (rmse))
    #print('MAPE: %.2f' % (mape))
    #print()
    return mae

def evaluate_average(net, test_loader, true_value, num_of_vertices, epoch):
    prediction = predict(net, test_loader)
    prediction = (prediction.transpose((0, 2, 1))
                  .reshape(prediction.shape[0], -1))
    
    #fast test only in MAE result by 12 Tp
    mae_t = 0
    rmse_t = 0 
    mape_t = 0
    
    for i in range(1,2):
    #for i in [1]:

        print('current epoch: %s, predict %s points' % (epoch, i))

        mae = mean_absolute_error(true_value[:, : i * num_of_vertices],
                                  prediction[:, : i * num_of_vertices])
        rmse = mean_squared_error(true_value[:, : i * num_of_vertices],
                                  prediction[:, : i * num_of_vertices]) ** 0.5
        mape = masked_mape_np(true_value[:, : i * num_of_vertices],
                              prediction[:, : i * num_of_vertices], 0)
        mae_t = mae_t + mae
        rmse_t = rmse_t +rmse
        mape_t = mape_t + mape

    mae_a = mae_t/1
    rmse_a = rmse_t/1
    mape_a = mape_t/1
    
    print('MAE: %.2f' % (mae_a))
    print('RMSE: %.2f' % (rmse_a))
    print('MAPE: %.2f' % (mape_a))
    print()
    return mae_a,rmse_a,mape_a