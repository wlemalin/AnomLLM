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
        )
    }
