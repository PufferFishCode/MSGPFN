#!/usr/bin/env python
# coding: utf-8

# # Ideas: 
# 
#    ## use MoE 
#         - only GP priors (vary generalization / fitting degree)
#             - much noise -> high generalization 
#             - little noise -> low generalization
#             - small length scale -> low generalization/ overfitting
#             - high length scale -> high generalization/ underfitting
#             
#         - only MLP priors (vary "expressiveness" of model)
#             - increase mlp hidden dim -> increase expressiveness
#             - decreas mlp hidden dim -> decrease expressiveness
#             - increase dropout prob -> generalization
#             - decrease dropout prob -> overfitting 
#             - change number of causes -> increase / decrease expressiveness
#         - mixel bag priors
#             - vary overfitting / generalization 
#             
#    ## how to choose the different configurations
#         - random choice 
#         - Bayesian optimization (how?)
#         - multi fidelity

# In[1]:


#%load_ext autoreload

#%autoreload 2


# In[2]:


import random
import time
import warnings
from datetime import datetime

import torch

import numpy as np

import matplotlib.pyplot as plt
from scripts.differentiable_pfn_evaluation import eval_model_range
from scripts.model_builder import get_model, get_default_spec, save_model, load_model
from scripts.transformer_prediction_interface import transformer_predict, get_params_from_config, load_model_workflow


from datasets import load_openml_list, open_cc_dids, open_cc_valid_dids
from priors.utils import plot_prior, plot_features
from priors.utils import uniform_int_sampler_f

from scripts.tabular_metrics import calculate_score_per_method, calculate_score
from scripts.tabular_evaluation import evaluate

from priors.differentiable_prior import DifferentiableHyperparameterList, draw_random_style, merge_style_with_info
from scripts import tabular_metrics
from notebook_utils import *

from copy import deepcopy
from tabpfn.priors.differentiable_prior import replace_differentiable_distributions
from ConfigSpace import hyperparameters as CSH
import ConfigSpace as CS


# In[3]:


large_datasets = True
max_samples = 10000 if large_datasets else 5000
bptt = 10000 if large_datasets else 3000
suite='cc'


# In[4]:


device = 'cpu'
base_path = '.'
max_features = 100


# In[5]:


def print_models(model_string):
    print(model_string)

    for i in range(80):
        for e in range(50):
            exists = Path(os.path.join(base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt')).is_file()
            if exists:
                print(os.path.join(base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt'))
        print()


# In[6]:


def train_function(config_sample, i, add_name=''):
    start_time = time.time()
    N_epochs_to_save = 50
    
    def save_callback(model, epoch):
        if not hasattr(model, 'last_saved_epoch'):
            model.last_saved_epoch = 0
        if ((time.time() - start_time) / (maximum_runtime * 60 / N_epochs_to_save)) > model.last_saved_epoch:
            print('Saving model..')
            config_sample['epoch_in_training'] = epoch
            save_model(model, base_path, f'models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{model.last_saved_epoch}.cpkt',
                           config_sample)
            model.last_saved_epoch = model.last_saved_epoch + 1 # TODO: Rename to checkpoint
    
    model = get_model(config_sample
                      , device
                      , should_train=True
                      , verbose=1
                      , epoch_callback = save_callback)
    
    return


# In[7]:


def print_config(config, indent=0):
    sorted_config = dict(sorted(config.items(), key=lambda x: str(x[0])))
    for key, value in sorted_config.items():
        if isinstance(value, dict):
            print(f"{' ' * indent}{key}:")
            print_config(value, indent + 4)
        else:
            print(f"{' ' * indent}{key}: {value}")


# # Create Hyperparameters for Priors

# In[8]:


def get_prior_config(config_type, causal_config = None, gp_config = None, bnn_config= None):
    if config_type == 'causal':
        return get_prior_config_causal(causal_config=causal_config)
    elif config_type == 'gp':
        return get_prior_config_gp(gp_config=gp_config)
    elif config_type == 'bnn':
        return get_prior_config_bnn(bnn_config=bnn_config)


# In[9]:


def get_prior_config_causal( causal_config, max_features=100):
    config_general = get_general_config(max_features, 50, eval_positions=[30])
    config_general_real_world = {**config_general}

    config_flexible_categorical = get_flexible_categorical_config(max_features)
    config_flexible_categorical_real_world = {**config_flexible_categorical}
    config_flexible_categorical_real_world[
        'num_categorical_features_sampler_a'] = -1.0  # Categorical features disabled by default

    config_gp = {}
    config_mlp = {}

    config_diff = get_diff_config(causal_config=causal_config)

    config = {**config_general_real_world, **config_flexible_categorical_real_world, **config_diff, **config_gp,
              **config_mlp}

    return config


# In[10]:


def get_prior_config_gp(gp_config, max_features=100):
    config_general = get_general_config(max_features, 50, eval_positions=[30])
    config_general_real_world = {**config_general}
    
    config_flexible_categorical = get_flexible_categorical_config(max_features)
    config_flexible_categorical_real_world = {**config_flexible_categorical}
    
    config_gp = {}

    config_diff = get_diff_config(gp_config=gp_config)
    
    config = {**config_general_real_world, **
              config_flexible_categorical_real_world, **config_diff, **config_gp}
    config['differentiable_hyperparameters']['prior_bag_exp_weights_1'] = {'distribution': 'uniform', 'min': 0.0,
                                                                           'max': .01}  # Never select MLP
    return config


# In[11]:


def get_prior_config_bnn(bnn_config, max_features=100):
    config_general = get_general_config(max_features, 50, eval_positions=[30])
    config_general_real_world = {**config_general}

    config_flexible_categorical = get_flexible_categorical_config(max_features)
    config_flexible_categorical_real_world = {**config_flexible_categorical}

    config_gp = {}
    config_mlp = {}

    config_diff = get_diff_config(bnn_config=bnn_config)

    config = {**config_general_real_world, **config_flexible_categorical_real_world, **config_diff, **config_gp,
              **config_mlp}

    config['differentiable_hyperparameters']['prior_bag_exp_weights_1'] = {'distribution': 'uniform',
                                                                           'min': 1000.0,
                                                                           'max': 1001.0}  # Always select MLP
    return config


# In[12]:


def get_general_config(max_features, bptt, eval_positions=None):
    """"
    Returns the general PFN training hyperparameters.
    """
    config_general = {
        "lr": CSH.UniformFloatHyperparameter('lr', lower=0.0001, upper=0.00015, log=True),
        "dropout": CSH.CategoricalHyperparameter('dropout', [0.0]),
        # upper bound is -1
        "emsize": CSH.CategoricalHyperparameter('emsize', [2 ** i for i in range(8, 9)]),
        "batch_size": CSH.CategoricalHyperparameter('batch_size', [2 ** i for i in range(6, 8)]),
        "nlayers": CSH.CategoricalHyperparameter('nlayers', [12]),
        "num_features": max_features,
        "nhead": CSH.CategoricalHyperparameter('nhead', [4]),
        "nhid_factor": 2,
        "bptt": bptt,
        "eval_positions": None,
        "seq_len_used": bptt,
        # hp.choice('sampling', ['mixed', 'normal']), # uniform
        "sampling": 'normal',
        "epochs": 80,
        "num_steps": 100,
        "verbose": False,
        "mix_activations": False,
        "pre_sample_causes": True,
        "multiclass_type": 'rank'
    }

    return config_general



# ## Causal Structural Model Hyperparameters
# 

# In[13]:


def get_diff_causal(num_layers_max_alpha=2,
                    num_layers_max_scale=3,
                    prior_mlp_hidden_dim_max_alpha=3,
                    prior_mlp_hidden_dim_max_scale=100,
                    prior_mlp_dropout_prob_scale=0.6,
                    prior_mlp_dropout_prob_min=0.1,
                    prior_mlp_dropout_prob_max=5.0,
                    noise_std_max_mean=0.3,
                    noise_std_min_mean=0.0001, 
                    init_std_max_mean=10.0,
                    init_std_min_mean=0.01,
                    num_causes_max_alpha=3, 
                    num_causes_max_scale=7):
    """"
    Returns the configuration parameters for a differentiable wrapper around MLP / Causal mixture.
    """
    diff_causal = {
        # "mix_activations": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        # "num_layers": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 6, 'min_mean': 1, 'round': True,
        #               'lower_bound': 2},
        "num_layers": {'distribution': 'meta_gamma',
                       'max_alpha': num_layers_max_alpha,
                       'max_scale': num_layers_max_scale,
                       'round': True,
                       'lower_bound': 2},
        # Better beta?
        # "prior_mlp_hidden_dim": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 130, 'min_mean': 5,
        #                         'round': True, 'lower_bound': 4},
        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma',
                                 'max_alpha': 3,
                                 'max_scale': 100,
                                 'round': True,
                                 'lower_bound': 4},

        "prior_mlp_dropout_prob": {'distribution':
                                   'meta_beta',
                                   'scale': 0.6,
                                   'min': 0.1,
                                   'max': 5.0},
        # This mustn't be too high since activations get too large otherwise

        "noise_std": {'distribution': 'meta_trunc_norm_log_scaled',
                      'max_mean': .3,
                      'min_mean': 0.0001,
                      'round': False,
                      'lower_bound': 0.0},

        "init_std": {'distribution': 'meta_trunc_norm_log_scaled',
                     'max_mean': 10.0,
                     'min_mean': 0.01,
                     'round': False,
                     'lower_bound': 0.0},

        # "num_causes": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 12, 'min_mean': 1, 'round': True,
        #               'lower_bound': 1},
        "num_causes": {'distribution': 'meta_gamma',
                       'max_alpha': 3,
                       'max_scale': 7,
                       'round': True,
                       'lower_bound': 2},

        "is_causal": {'distribution': 'meta_choice',
                      'choice_values': [True, False]},

        "pre_sample_weights": {'distribution': 'meta_choice',
                               'choice_values': [True, False]},

        "y_is_effect": {'distribution': 'meta_choice',
                        'choice_values': [True, False]},

        "sampling": {'distribution': 'meta_choice',
                     'choice_values': ['normal', 'mixed']},

        "prior_mlp_activations": {'distribution': 'meta_choice_mixed',
                                  'choice_values': [torch.nn.Tanh, torch.nn.Identity, torch.nn.ReLU]},

        "block_wise_dropout": {'distribution': 'meta_choice',
                               'choice_values': [True, False]},

        "sort_features": {'distribution': 'meta_choice',
                          'choice_values': [True, False]},

        "in_clique": {'distribution': 'meta_choice',
                      'choice_values': [True, False]},

        # 'pre_sample_causes': {'distribution': 'meta_choice', 'choice_values': [True, False]},
    }

    return diff_causal



# ## Gaussian Process Hyperparameters

# In[14]:


def get_diff_gp(os_max_mean=10,
                os_min_mean=0.00001,
                ls_max_mean=10, 
                ls_min_mean=0.00001, 
                noise_choices = [0.00001, 0.0001, 0.01]):
    """"
    Returns the configuration parameters for a differentiable wrapper around GP.
    """
    diff_gp = {
        'outputscale': {'distribution': 'meta_trunc_norm_log_scaled',
                        'max_mean': os_max_mean,
                        'min_mean': os_min_mean,
                        'round': False,
                        'lower_bound': 0},
        'lengthscale': {'distribution': 'meta_trunc_norm_log_scaled',
                        'max_mean': ls_max_mean,
                        'min_mean': ls_min_mean,
                        'round': False,
                        'lower_bound': 0},
        'noise': {'distribution': 'meta_choice',
                  'choice_values': noise_choices}
    }

    return diff_gp


# In[15]:


def get_flexible_categorical_config(max_features):
    """"
    Returns the configuration parameters for the tabular multiclass wrapper.
    """
    config_flexible_categorical = {
        "nan_prob_unknown_reason_reason_prior": CSH.CategoricalHyperparameter('nan_prob_unknown_reason_reason_prior', [0.5]),
        "categorical_feature_p": CSH.CategoricalHyperparameter('categorical_feature_p', [0.0, 0.1, 0.2]),
        "nan_prob_no_reason": CSH.CategoricalHyperparameter('nan_prob_no_reason', [0.0, 0.1]),
        "nan_prob_unknown_reason": CSH.CategoricalHyperparameter('nan_prob_unknown_reason', [0.0]),
        "nan_prob_a_reason": CSH.CategoricalHyperparameter('nan_prob_a_reason', [0.0]),
        # "num_classes": lambda : random.randint(2, 10), "balanced": False,
        "max_num_classes": 2,
        "num_classes": 2,
        # NN
        "noise_type": CSH.CategoricalHyperparameter('noise_type', ["Gaussian"]),
        "balanced": True,
        "normalize_to_ranking": CSH.CategoricalHyperparameter('normalize_to_ranking', [False]),
        "set_value_to_nan": CSH.CategoricalHyperparameter('set_value_to_nan', [0.5, 0.2, 0.0]),
        "normalize_by_used_features": True,
        "num_features_used":
            {'uniform_int_sampler_f(3,max_features)': uniform_int_sampler_f(
                1, max_features)}
        # hp.choice('conv_activation', [{'distribution': 'uniform', 'min': 2.0, 'max': 8.0}, None]),
    }
    return config_flexible_categorical



# In[16]:


def get_diff_flex():
    """"
    Returns the configuration parameters for a differentiable wrapper around the tabular multiclass wrapper.
    """
    diff_flex = {
        # "ordinal_pct": {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
        # "num_categorical_features_sampler_a": hp.choice('num_categorical_features_sampler_a',
        #                                                 [{'distribution': 'uniform', 'min': 0.3, 'max': 0.9}, None]),
        # "num_categorical_features_sampler_b": {'distribution': 'uniform', 'min': 0.3, 'max': 0.9},

        # CSH.CategoricalHyperparameter('output_multiclass_ordered_p', [0.0, 0.1, 0.2]),
        "output_multiclass_ordered_p": {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
        "multiclass_type": {'distribution': 'meta_choice', 'choice_values': ['value', 'rank']},
    }

    return diff_flex


# In[17]:


def get_diff_prior_bag(dist_type="uniform", weights_min=2.0, weights_max=10.0):
    """"
    Returns the configuration parameters for a GP and MLP / Causal mixture.
    """
    diff_prior_bag = {
        'prior_bag_exp_weights_1': {'distribution': dist_type, 'min': weights_min, 'max': weights_max},
        # MLP Weight (Biased, since MLP works better, 1.0 is weight for prior number 0)
    }

    return diff_prior_bag


# In[18]:


def get_diff_config(prior_bag_config = None, causal_config=None, gp_config = None, flex_config = None):
    """"
    Returns the configuration parameters for a differentiable wrapper around GP and MLP / Causal mixture priors.
    """
    if prior_bag_config == None:
        diff_prior_bag = get_diff_prior_bag()
    else: 
        diff_prior_bag = get_diff_prior_bag(dist_type=prior_bag_config["dist_type"], 
                                            weights_min=prior_bag_config["weights_min"],
                                            weights_max=prior_bag_config["weights_max"]) 
        
    # --------------------------------------------------
    if causal_config == None:
        diff_causal = get_diff_causal()
    else:
        diff_causal = get_diff_causal(num_layers_max_alpha=causal_config["num_layers_max_alpha"],
                    num_layers_max_scale=causal_config["num_layers_max_scale"],
                    prior_mlp_hidden_dim_max_alpha=causal_config["prior_mlp_hidden_dim_max_alpha"],
                    prior_mlp_hidden_dim_max_scale=causal_config["prior_mlp_hidden_dim_max_scale"],
                    prior_mlp_dropout_prob_scale=causal_config["prior_mlp_dropout_prob_scale"],
                    prior_mlp_dropout_prob_min=causal_config["prior_mlp_dropout_prob_min"],
                    prior_mlp_dropout_prob_max=causal_config["prior_mlp_dropout_prob_max"],
                    noise_std_max_mean=causal_config["noise_std_max_mean"],
                    noise_std_min_mean=causal_config["noise_std_min_mean"], 
                    init_std_max_mean=causal_config["init_std_max_mean"],
                    init_std_min_mean=causal_config["init_std_min_mean"],
                    num_causes_max_alpha=causal_config["num_causes_max_alpha"], 
                    num_causes_max_scale=causal_config["num_causes_max_scale"]) # todo
        
    # --------------------------------------------------
    if gp_config == None:
        diff_gp = get_diff_gp()
        print(f"get diff config: gp_config is None")
    else: 
        print(f"get diff config: gp_config is not None")
        diff_gp = get_diff_gp(os_max_mean= gp_config["os_max_mean"],
                            os_min_mean=gp_config["os_min_mean"], 
                            ls_max_mean=gp_config["ls_max_mean"], 
                            ls_min_mean=gp_config["ls_min_mean"], 
                            noise_choices=gp_config["noise_choices"])
        
    # --------------------------------------------------
    if flex_config == None:
        diff_flex = get_diff_flex()
    else: 
        diff_flex = get_diff_flex() # todo
        
    # --------------------------------------------------
    config_diff = {'differentiable_hyperparameters': {
        **diff_prior_bag, **diff_causal, **diff_gp, **diff_flex}}

    return config_diff


# In[19]:


def reload_config(config_type='causal',
                  causal_config=None,
                  gp_config=None,
                  bnn_config=None, 
                  task_type='multiclass', 
                  longer=0):
    print(f"{gp_config} --- 1")
    config = get_prior_config(config_type=config_type, 
                              causal_config=causal_config,
                              gp_config=gp_config,
                              bnn_config= bnn_config) 
    
    config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    
    model_string = ''
    
    config['epochs'] = 12000
    config['recompute_attn'] = True

    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    return config, model_string


# # Sample Hyperparameters for Priors

# In[20]:


def list_all_hps_in_nested(config):
    """"
    Returns a list of hyperparameters from a nested dict of hyperparameters.
    """

    if isinstance(config, CSH.Hyperparameter):
        return [config]
    elif isinstance(config, dict):
        result = []
        for k, v in config.items():
            result += list_all_hps_in_nested(v)
        return result
    else:
        return []


# In[21]:


def create_configspace_from_hierarchical(config):
    cs = CS.ConfigurationSpace()
    for hp in list_all_hps_in_nested(config):
        cs.add_hyperparameter(hp)
    return cs


# In[22]:


def fill_in_configsample(config, configsample):
    # config is our dict that defines config distribution
    # configsample is a CS.Configuration
    hierarchical_configsample = deepcopy(config)
    for k, v in config.items():
        if isinstance(v, CSH.Hyperparameter):
            hierarchical_configsample[k] = configsample[v.name]
        elif isinstance(v, dict):
            hierarchical_configsample[k] = fill_in_configsample(v, configsample)
    return hierarchical_configsample


# In[23]:


def evaluate_hypers(config, sample_diff_hps=False):
    """"
    Samples a hyperparameter configuration from a sampleable configuration (can be used in HP search).
    """
    if sample_diff_hps:
        # I do a deepcopy here, such that the config stays the same and can still be used with diff. hps
        config = deepcopy(config)
        replace_differentiable_distributions(config)
    cs = create_configspace_from_hierarchical(config)
    cs_sample = cs.sample_configuration()
    return fill_in_configsample(config, cs_sample)


# ## Start configuration creation

# In[24]:


def sample_gp_config_meta():
    ## Sathya
    #result = {"os_max_mean": 6, 
    #            "os_min_mean":0.001, 
    #            "ls_max_mean":6,
    #            "ls_min_mean":0.0001,
    #            "noise_choices":[0.0001, 0.001, 0.1],
    #            }
    #
    # # 
    #Magnus
    results = {"os_max_mean": 10, 
                "os_min_mean":2, 
                "ls_max_mean":12,
                "ls_min_mean":4,
                "noise_choices":[0.0001, 0.0001, 0.001]}
    # 
    #  
    # Jack
    # results = {"os_max_mean": 15, 
    #             "os_min_mean":0.0001, 
    #             "ls_max_mean":12,
    #             "ls_min_mean":0.0001,
    #             "noise_choices":[0.0001, 0.0001, 0.001, 0.01, 0.1],
    #             
    # }
    #     
    # Ali
    # results = {"os_max_mean": 3, 
    #             "os_min_mean":0.0001, 
    #             "ls_max_mean":4,
    #             "ls_min_mean":0.0001,
    #             "noise_choices":[0.0001, 0.0001, 0.001, 0.01, 0.1],
    #            
    # }
    return results


# In[25]:


def sample_causal_config_meta(number_of_configs = 1):
    config_space = CS.ConfigurationSpace()

    num_layers_max_alpha = CSH.NormalFloatHyperparameter('num_layers_max_alpha', mu=2., sigma=0.2, log=False)#?? todo
    num_layers_max_scale = CSH.NormalFloatHyperparameter('num_layers_max_scale', mu=3, sigma=0.2, log=False)#?? todo
    
    prior_mlp_hidden_dim_max_alpha = CSH.NormalFloatHyperparameter('prior_mlp_hidden_dim_max_alpha', mu=3, sigma=0.2, log=False)#?? todo
    prior_mlp_hidden_dim_max_scale = CSH.NormalFloatHyperparameter('prior_mlp_hidden_dim_max_scale', mu=100, sigma=0.2, log=False)#?? todo
    
    prior_mlp_dropout_prob_scale = CSH.NormalFloatHyperparameter('prior_mlp_dropout_prob_scale', mu=0.6, sigma=0.2, log=False)#?? todo
    prior_mlp_dropout_prob_min = CSH.NormalFloatHyperparameter('prior_mlp_dropout_prob_min', mu=0.1, sigma=0.2, log=False)#?? todo
    prior_mlp_dropout_prob_max = CSH.NormalFloatHyperparameter('prior_mlp_dropout_prob_max', mu=5, sigma=0.2, log=False)#?? todo
    
    noise_std_max_mean = CSH.NormalFloatHyperparameter('noise_std_max_mean', mu=0.3, sigma=0.2, log=False)#?? todo
    noise_std_min_mean = CSH.NormalFloatHyperparameter('noise_std_min_mean', mu=0.0001, sigma=0.2, log=False)#?? todo
    
    init_std_max_mean = CSH.NormalFloatHyperparameter('init_std_max_mean', mu=10.0, sigma=0.2, log=False)#?? todo
    init_std_min_mean = CSH.NormalFloatHyperparameter('init_std_min_mean', mu=0.01, sigma=0.2, log=False)#?? todo
    
    num_causes_max_alpha = CSH.NormalFloatHyperparameter('num_causes_max_alpha', mu=3, sigma=0.2, log=False)#?? todo
    num_causes_max_scale = CSH.NormalFloatHyperparameter('num_causes_max_scale', mu=7, sigma=0.2, log=False)#?? todo

    config_space.add_hyperparameters([num_layers_max_alpha, 
                                      num_layers_max_scale, 
                                      prior_mlp_hidden_dim_max_alpha, 
                                      prior_mlp_hidden_dim_max_scale, 
                                      prior_mlp_dropout_prob_scale, 
                                      prior_mlp_dropout_prob_min, 
                                      prior_mlp_dropout_prob_max, 
                                      noise_std_max_mean, 
                                      noise_std_min_mean, 
                                      init_std_max_mean, 
                                      init_std_min_mean,
                                      num_causes_max_alpha, 
                                      num_causes_max_scale])
    
    config_space_samples =config_space.sample_configuration(number_of_configs) 
    
    # cast to list if only one configuration to handle it everytime equally
    config_space_samples = [config_space_samples] if number_of_configs <= 1 else config_space_samples
    
    results = []
    for config_space_sample in config_space_samples:
        config_space_sample = config_space_sample.get_dictionary()
        results.append(config_space_sample)
    
    return results
                     


# In[ ]:


causal_configs = sample_causal_config_meta() if False else None
bnn_config = None
gp_configs = sample_gp_config_meta() 


causal_config = causal_configs if causal_configs else None
gp_config = gp_configs if gp_configs else None

config, model_string = reload_config(config_type='gp',
                                     longer=1,
                                     causal_config=causal_config, 
                                     gp_config=gp_config, 
                                     bnn_config = bnn_config)
config['bptt_extra_samples'] = None
# diff
config['output_multiclass_ordered_p'] = 0.
del config['differentiable_hyperparameters']['output_multiclass_ordered_p']
config['multiclass_type'] = 'rank'
del config['differentiable_hyperparameters']['multiclass_type']
config['sampling'] = 'normal' # vielleicht schlecht?
del config['differentiable_hyperparameters']['sampling']
config['pre_sample_causes'] = True
# end diff
config['multiclass_loss_type'] = 'nono' # 'compatible'
config['normalize_to_ranking'] = False # False
config['categorical_feature_p'] = .2 # diff: .0
# turn this back on in a random search!?
config['nan_prob_no_reason'] = .0
config['nan_prob_unknown_reason'] = .0 # diff: .0
config['set_value_to_nan'] = .1 # diff: 1.
config['normalize_with_sqrt'] = False
config['new_mlp_per_example'] = True
config['prior_mlp_scale_weights_sqrt'] = True
config['batch_size_per_gp_sample'] = None
config['normalize_ignore_label_too'] = False
config['differentiable_hps_as_style'] = False
config['max_eval_pos'] = 1000
config['random_feature_rotation'] = True
config['rotate_normalized_labels'] = True
config["mix_activations"] = False # False heisst eig True
config['emsize'] = 512
config['nhead'] = config['emsize'] // 128
config['bptt'] = 1024+128
config['canonical_y_encoder'] = False
config['aggregate_k_gradients'] = 8
config['batch_size'] = 8*config['aggregate_k_gradients']
config['num_steps'] = 1024//config['aggregate_k_gradients']
config['epochs'] = 400
config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...
config['train_mixed_precision'] = True
config['efficient_eval_masking'] = True
#print_config(config)
config_sample = evaluate_hypers(config)
config_sample['batch_size'] = 4
# print_config(config_sample)
if True: 
    model = get_model(config_sample, device, should_train=True, verbose=0)
    save_model(model, base_path, f'baseline_model_gp_1.cpkt', config_sample)

