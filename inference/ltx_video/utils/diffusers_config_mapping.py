# Source Generated with Decompyle++
# File: diffusers_config_mapping.pyc (Python 3.10)


def make_hashable_key(dict_key):
    
    def convert_value(value = None):
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, dict):
            return dict((k, convert_value(v)) for k, v in value.items())

    return dict((k, convert_value(v)) for k, v in dict_key.items())

DIFFUSERS_SCHEDULER_CONFIG = {
    '_class_name': 'FlowMatchEulerDiscreteScheduler',
    '_diffusers_version': '0.32.0.dev0',
    'base_image_seq_len': 1024,
    'base_shift': 0.95,
    'invert_sigmas': False,
    'max_image_seq_len': 4096,
    'max_shift': 2.05,
    'num_train_timesteps': 1000,
    'shift': 1,
    'shift_terminal': 0.1,
    'use_beta_sigmas': False,
    'use_dynamic_shifting': True,
    'use_exponential_sigmas': False,
    'use_karras_sigmas': False }
# WARNING: Decompyle incomplete
