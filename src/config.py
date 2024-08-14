from prompt import create_openai_request


def create_batch_api_configs():
    return {
        '1shot-vision': lambda series, train_dataset: create_openai_request(
            series,
            vision=True,
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
        )
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
