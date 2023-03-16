import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
choice_mapping = dict(  all_choice = pd.Series(True, index=choices.index),
                        previous_choice_l = choices == 0,  # For before go cue, since it is already offseted, this becomes the previous trial; for after go cue or iti start, this is the trial that just happened
                        previous_choice_r = choices == 1,
                        next_choice_l = (choices == 0).shift(-1).fillna(False),  # choice immediately AFTER 'before go cue', but the choice of the next trial of "after go cue" / "iti start" etc.
                        next_choice_r = (choices == 1).shift(-1).fillna(False))
'''

def compute_group_tuning(df_unit_latent_bin_firing, 
                         unit_keys=None,
                         if_z_score_firing=True,   # z-score
                         significance_level=None,  # if None, all cells
                         if_flip_tuning=True, # auto flip here, within the selected unit_keys, where users controls significance etc.
                         choice_group='all_choice'
                         ):
    
    if unit_keys is None:
        unit_keys = df_unit_latent_bin_firing.index
        
    if significance_level is not None:
        unit_keys = unit_keys[df_unit_latent_bin_firing.loc[unit_keys].p < significance_level]
    
    selected_tuning = df_unit_latent_bin_firing.loc[unit_keys, choice_group]['mean']
    z_mean = df_unit_latent_bin_firing.loc[unit_keys, 'z_mean']
    z_std = df_unit_latent_bin_firing.loc[unit_keys, 'z_std']
    
    if if_z_score_firing:
        #  selected_tuning = selected_tuning.apply(scipy.stats.zscore, axis=1, nan_policy='omit')  # using mean and binned activity (not full dynamic range of the neuron!)
        selected_tuning = selected_tuning.sub(z_mean, axis=0).div(z_std, axis=0)       # using dynamic range of the neuron (average psth aligned to go_cue)
        
    if if_flip_tuning:
        to_flip = df_unit_latent_bin_firing['r'] < 0
        selected_tuning.loc[to_flip] = np.array(selected_tuning.loc[to_flip])[:, ::-1]

    tuning_mean = selected_tuning.mean(axis=0)
    tuning_sem = selected_tuning.sem(axis=0)      
    
    return tuning_mean, tuning_sem, selected_tuning



   