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