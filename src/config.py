from prompt import create_openai_request


def create_batch_api_configs():
    return {
        '1shot-vision': lambda series, train_dataset: create_openai_request(
            series,
            vision=True,
            few_shots=train_dataset.few_shots(num_shots=1)
        ),
        '0shot-vision-cot': lambda series, train_dataset: create_openai_request(
            series,
            vision=True,
            cot=train_dataset.name,
            few_shots=train_dataset.few_shots(num_shots=0)
        ),
        '1shot-vision-cot': lambda series, train_dataset: create_openai_request(
            series,
            vision=True,
            cot=train_dataset.name,
            few_shots=train_dataset.few_shots(num_shots=1)
        ),
        '0shot-vision': lambda series, train_dataset: create_openai_request(
            series,
            vision=True,
            few_shots=train_dataset.few_shots(num_shots=0)
        ),
        '1shot-text': lambda series, train_dataset: create_openai_request(
            series,
            vision=False,
            few_shots=train_dataset.few_shots(num_shots=1)
        ),
        '0shot-text': lambda series, train_dataset: create_openai_request(
            series,
            vision=False,
            few_shots=train_dataset.few_shots(num_shots=0)
        ),
        '0shot-text-s0.3': lambda series, train_dataset: create_openai_request(
            series,
            vision=False,
            few_shots=train_dataset.few_shots(num_shots=1),
            series_args={'scale': 0.3}
        ),
        '1shot-text-s0.3': lambda series, train_dataset: create_openai_request(
            series,
            vision=False,
            few_shots=train_dataset.few_shots(num_shots=1),
            series_args={'scale': 0.3}
        ),
        '0shot-text-s0.3-cot': lambda series, train_dataset: create_openai_request(
            series,
            vision=False,
            few_shots=train_dataset.few_shots(num_shots=0),
            series_args={'scale': 0.3},
            cot=train_dataset.name
        ),
        '1shot-text-s0.3-cot': lambda series, train_dataset: create_openai_request(
            series,
            vision=False,
            few_shots=train_dataset.few_shots(num_shots=1),
            series_args={'scale': 0.3},
            cot=train_dataset.name
        ),
    }


def scale_result_str(input_string, scale=0.3):
    import re
    
    def replace_func(match):
        integer = int(match.group())
        return str(int(integer / scale))
    
    return re.sub(r'\d+', replace_func, input_string)


def postprocess_configs():
    return {
        '0shot-text-s0.3': lambda s: scale_result_str(s, 0.3),
        '1shot-text-s0.3': lambda s: scale_result_str(s, 0.3)
    }


def dataset_descriptions():
    return {
        "trend": {
            "normal": "the normal data follows a steady but slowly increasing trend from -1 to 1",
            "abnormal": "the data appears to either increase much faster or decrease, deviating from the normal trend",
            "abnormal_summary": "trend or speed changes"
        },
        "point": {
            "normal": "the normal data is a periodic sine wave between -1 and 1",
            "abnormal": "the data appears to become noisy and unpredictable, deviating from the normal periodic pattern",
            "abnormal_summary": "noises"
        },
        "freq": {
            "normal": "the normal data is a periodic sine wave between -1 and 1",
            "abnormal": "the data suddenly changes frequency, with very different periods between peaks",
            "abnormal_summary": "frequency changes"
        },
        "range": {
            "normal": "the normal data appears to be Gaussian noise with mean 0",
            "abnormal": "the data suddenly encounter spikes, with values much further from 0 than the normal noise",
            "abnormal_summary": "amplitude changes"
        }
    }
