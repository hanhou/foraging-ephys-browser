import plotly.express as px


# Constants
all_models =  [
                # split rpe to rew and chQ
               'dQ, sumQ, rew, chQ',              
               'dQ, sumQ, rew, chQ, C*2',
               'dQ, sumQ, rew, chQ, C*2, t',
               'dQ, sumQ, rew, chQ, C*2, R*1',
               'dQ, sumQ, rew, chQ, C*2, R*1, t',
               'dQ, sumQ, rew, chQ, C*2, R*5, t',
               'dQ, sumQ, rew, chQ, C*2, R*10, t',            
               'contraQ, ipsiQ, rew, chQ',
               'contraQ, ipsiQ, rew, chQ, C*2, R*5, t',
    
                # original rpes
               'dQ, sumQ, rpe', 
               'dQ, sumQ, rpe, C*2', 
               'dQ, sumQ, rpe, C*2, t', 
               'dQ, sumQ, rpe, C*2, R*1', 
               'dQ, sumQ, rpe, C*2, R*1, t',
               'dQ, sumQ, rpe, C*2, R*5, t',
               'dQ, sumQ, rpe, C*2, R*10, t',
               'contraQ, ipsiQ, rpe',
               'contraQ, ipsiQ, rpe, C*2, R*5, t',
               ]


model_name_mapper = {model:  ' + '.join([{'dQ': 'dQ', ' sumQ': 'sumQ', ' rpe': 'rpe', 'contraQ': 'contraQ', ' ipsiQ': 'ipsiQ',
                                        ' C*2': 'choice', ' t': 'trial#', 
                                        ' R*1': 'firing back 1', ' R*5': 'firing back 5', ' R*10': 'firing back 10',
                                        ' rew': 'rew', ' chQ': 'chosenQ', 
                                        }[var] for var in model.split(',')])
                    for model in all_models}

model_color_mapper = {model:color for model, color in zip(all_models, px.colors.qualitative.Plotly * 3)}

period_name_mapper = {'before_2': 'Before GO (2s)', 'delay': 'Delay (median 60 ms)', 'go_1.2': 'After GO (1.2s)', 'go_to_end': 'GO to END', 
                  'iti_all': 'ITI (all, median 3.95s)', 'iti_first_2': 'ITI (first 2s)', 'iti_last_2': 'ITI (last 2s)'}

para_name_mapper = {'relative_action_value_ic': 'dQ', 
                    'total_action_value': 'sumQ', 
                    'ipsi_action_value': 'ipsiQ',
                    'contra_action_value': 'contraQ', 
                    'rpe': 'rpe',
                    'choice_ic': 'choice (this)', 
                    'choice_ic_next': 'choice (next)',
                    'trial_normalized': 'trial number', 
                    'reward': 'reward',
                    'chosen_value': 'chosenQ',
                    **{f'firing_{n}_back': f'firing {n} back' for n in range(1, 11)},
                    }

# For significant proportion
sig_prop_vars = ['relative_action_value_ic', 'total_action_value', 
                'ipsi_action_value', 'contra_action_value', 
                'rpe', 'reward', 'chosen_value',
                'choice_ic', 'choice_ic_next', 'trial_normalized', 'firing_1_back',
                ]

sig_prop_color_mapping = {var: color for var, color in zip(sig_prop_vars, 
                                                           ['darkviolet', 'deepskyblue', 'darkblue', 'darkorange', 'gray', 'gray', 'black'] + 
                                                           px.colors.qualitative.Plotly_r)}



# For pure units
pure_unit_color_mapping =  {'pure_dQ': 'darkviolet',
                            'pure_sumQ': 'deepskyblue',
                            'pure_contraQ': 'darkblue',
                            'pure_ipsiQ': 'darkorange'}
                                
polar_classifiers = {'dQ, sumQ, rpe': [{'x_name': 'relative_action_value_ic', 'y_name': 'total_action_value'},
                                        {'pure_dQ': [(-22.5, 22.5), (-22.5 + 180, 180), (-180, -180 + 22.5)],
                                        'pure_sumQ': [(22.5 + 45, 67.5 + 45), (22.5 + 45 - 180, 67.5 + 45 - 180)],
                                        'pure_contraQ': [(22.5, 67.5), (22.5 - 180, 67.5 - 180)],
                                        'pure_ipsiQ': [(22.5 + 90, 67.5 + 90), (22.5 + 90 - 180, 67.5 + 90 - 180)]}],
                        
                     'contraQ, ipsiQ, rpe':  [{'x_name': 'ipsi_action_value', 'y_name': 'contra_action_value'},
                                            {'pure_dQ': [(22.5 + 90, 67.5 + 90), (22.5 + 90 - 180, 67.5 + 90 - 180)],
                                            'pure_sumQ': [(22.5, 67.5), (22.5 - 180, 67.5 - 180)],
                                            'pure_contraQ': [(22.5 + 45, 67.5 + 45), (22.5 + 45 - 180, 67.5 + 45 - 180)],
                                            'pure_ipsiQ': [(-22.5, 22.5), (-22.5 + 180, 180), (-180, -180 + 22.5)],}]
}

polar_classifiers['dQ, sumQ, rew, chQ'] = polar_classifiers['dQ, sumQ, rpe']
polar_classifiers['contraQ, ipsiQ, rew, chQ'] = polar_classifiers['contraQ, ipsiQ, rpe']
