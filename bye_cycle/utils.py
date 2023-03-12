import beep
from beep.structure.cli import auto_load_processed
import pandas as pd
import numpy as np
from tqdm import tqdm


def interpol_cycle(cell_cycle_data, columns=['voltage', 'current', 'cycle_index'], n_points=10, interpol_kind='linear'):
    from scipy.interpolate import interp1d
    interpol_cell_cycle_data = []
    for i,c in enumerate(columns):
        x = np.arange(cell_cycle_data[c].shape[0])
        f_out = interp1d(x, np.array([cell_cycle_data[c]]), kind=interpol_kind)
        new_x = np.linspace(0,cell_cycle_data[c].shape[0]-1, n_points)
        interpol_y = f_out(new_x)
        interpol_cell_cycle_data.append(interpol_y[0])
    return np.array(interpol_cell_cycle_data).T


def clean_cycle_data(cell_data, cycle_number, columns=['voltage', 'current', 'cycle_index', 'discharge_capacity']):
    selected_cell_data = cell_data.structured_data[columns].dropna()
    # Getting rid of 0 dishcarge data
    clean_cell_data_cycle_number = selected_cell_data.loc[(selected_cell_data['cycle_index'] == cycle_number) 
                                              & (selected_cell_data['discharge_capacity']!=0)]
    return clean_cell_data_cycle_number[columns]


def prep_features_per_cell(cell, n_points=100, columns=['voltage', 'current', 'cycle_index'], interpol_kind='linear'):
    max_cycle_idx = max(cell.structured_data['cycle_index'])
    interp_clean_cell = []
    empty_cycles = []
    for i in tqdm(range(max_cycle_idx)):
        clean_dat = clean_cycle_data(cell, i, columns=columns)
        # Looks like there are some missing cycle measurements so ...
        if clean_dat.shape[0] not in [0, 1, 2]:
            interp_clean_cell.append(interpol_cycle(clean_dat, n_points=n_points, columns=columns, interpol_kind=interpol_kind))
        else:
            empty_cycles.append(i)
    interp_clean_cell = np.array(interp_clean_cell)
    return interp_clean_cell


def find_slope(cycle_window_label):
    mid_window_index = len(cycle_window_label)//2
    low = np.array(cycle_window_label)[0]
    mid = np.array(cycle_window_label)[mid_window_index]
    high = np.array(cycle_window_label)[-1]
    slope_1 = (mid[1] - low[1]) / (mid[0] - low[0]) 
    slope_2 = (high[1] - mid[1]) / (high[0] - mid[0]) 
    return slope_1, slope_2


def find_renumbedred_index(cell):
    '''This function finds the renumbered index for the cycles that are followed by a diagnostic cycle'''
    renumbered_index_for_cycle_following_a_diagnostic_cycle = []
    i = 0
    for k,cycle in enumerate(cell):
        actual_cycle_index = np.unique(cycle['cycle_index'])[0]
        if actual_cycle_index != i:
            renumbered_index_for_cycle_following_a_diagnostic_cycle.append(k-1)
            i = actual_cycle_index
        i += 1
    return renumbered_index_for_cycle_following_a_diagnostic_cycle


def index_convoluter(cell, cycle_window_size=20, overlap_cycle_window=False,
                     overlap_size=5, skip_diagnistic_in_window=False):
    '''This function generates a nested list of indices by convolting the cycles based on cycle_window_size. 
        You can choose to have your window of cycles overlap, or skipp the windows that involve diagnostic cycles in between.
    '''
    max_cycle_index = len(cell) #30 #
    if cycle_window_size < overlap_size:
        raise ValueError(f'If overlapping, overlap_size ({overlap_size}) should be samller than cycle_window_size ({cycle_window_size}). Please check inputs.')
    if not cycle_window_size < max_cycle_index:
        raise ValueError(f'cycle_window_size ({cycle_window_size}) should be samller than maximum number of cycles in cell ({max_cycle_index}). Please check inputs.')
    if overlap_cycle_window:
        non_overlap =  cycle_window_size - overlap_size
    else:
        non_overlap = cycle_window_size
    intervals = range(0, max_cycle_index, non_overlap)
    if not skip_diagnistic_in_window:
        index_covolutions = [list(range(x, x + cycle_window_size)) for x in intervals if x <= max_cycle_index-cycle_window_size]
    else:
#         skipped_indices = [5, 6, 7, 20]
        skipped_indices = find_renumbedred_index(cell)
        index_covolutions = [list(range(x, x + cycle_window_size)) for x in intervals if x <= max_cycle_index-cycle_window_size
                             and not (set(list(range(x, x + cycle_window_size))[:-1]) & set(skipped_indices))]
    return index_covolutions


def make_feature_arrays(data):
    X_dt_values = []
    X_time_series = []
    for X in data:
        X_dt_values.append(X[1])
        X_time_series.append(X[0])
    return np.array(X_time_series), np.array(X_dt_values)


def make_label_arrays(data):
    slopes = []
    discharge_cap = []
    for Y in data:
        discharge_cap.append(Y[1])
        slopes.append(Y[0])
    return np.array(discharge_cap), np.array(slopes)


def scale(inputs, scaler=None):
    from sklearn import preprocessing
    inputs_shape = inputs.shape
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(inputs.reshape(-1, inputs_shape[-1]))
    scaled_input = scaler.transform(inputs.reshape(-1, inputs_shape[-1])).reshape(inputs_shape)
    return scaler, scaled_input


def process_data(cell_features, cell_labels, seed=0, train_size=0.9, val_size=0.2):
    prepared_data = list(zip(cell_features, cell_labels))
    import random
    random.seed(0)
    shuffled_indices = list(range(len(prepared_data)))
    random.shuffle(prepared_data)

    N_train = int(train_size *len(prepared_data))
    N_val = int(val_size * N_train)
    train_cells = prepared_data[:N_train]
    test_cells = prepared_data[N_train:]

    val_cells = train_cells[:N_val]
    train_cells = train_cells[N_val:]
    print(f'Train size:{len(train_cells)}\n'+ f'Validation size:{len(val_cells)}\n'+ f'Test size:{len(test_cells)}\n' + f'Total cells: {len(train_cells)+len(test_cells)+len(val_cells)}')
   
    X_train = [x for xs in train_cells for x in xs[0]]
    Y_train = [x for xs in train_cells for x in xs[1]]
    dc_train, slopes_train = make_label_arrays(Y_train)
    X_train_time_series, X_train_dt = make_feature_arrays(X_train)

    indices = list(range(len(X_train_time_series)))
    random.shuffle(indices)
    X_train_time_series = X_train_time_series[indices]
    X_train_dt = X_train_dt[indices]


    slopes_train = slopes_train[indices]
    dc_train = dc_train[indices]


    X_val = [x for xs in val_cells for x in xs[0]]
    Y_val = [x for xs in val_cells for x in xs[1]]
    dc_val, slopes_val = make_label_arrays(Y_val)
    X_val_time_series, X_val_dt = make_feature_arrays(X_val)

    X_test = [x for xs in test_cells for x in xs[0]]
    Y_test = [x for xs in test_cells for x in xs[1]]
    dc_test, slopes_test = make_label_arrays(Y_test)
    X_test_time_series, X_test_dt = make_feature_arrays(X_test)

    f_scaler, scaled_X_train_time_series = scale(X_train_time_series[...,1:])
    _ , scaled_X_val_time_series = scale(X_val_time_series[...,1:], f_scaler)
    _ , scaled_X_test_time_series = scale(X_test_time_series[...,1:], f_scaler)

    l_scaler, scaled_slopes_train = scale(slopes_train)
    _ , scaled_slopes_val = scale(slopes_val, l_scaler)
    _ , scaled_slopes_test = scale(slopes_test, l_scaler)

    return scaled_X_train_time_series, scaled_slopes_train, + \
            scaled_X_val_time_series, scaled_slopes_val, + \
            scaled_X_test_time_series, scaled_slopes_test, f_scaler, l_scaler

def linear_fit(new_x_point, slope, point_xy):
    return slope*(new_x_point-point_xy[0]) + point_xy[1]


def eval_dc_reconstruction(predicted_dc, ground_truth_dc):
    unique_keys, p_indices = np.unique(predicted_dc[:,0], return_index=True)
    unique_keys, g_indices = np.unique(ground_truth_dc[:,0], return_index=True)
    ground_truth_dc_size = ground_truth_dc.shape[0]
    return np.sqrt(((predicted_dc[p_indices][:,1] - ground_truth_dc[g_indices][:,1]) ** 2).mean())