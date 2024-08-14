import numpy as np
from scipy import interpolate
import json
import re


PROMPT = """Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list [].

Output template:
[{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
"""


LIMIT_PROMPT = "Assume there are up to 5 anomalies. "


SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


def scale_x_axis(data, scale_factor):
    """
    Scale the x-axis of a 1D numpy array.
    
    :param data: Input numpy array of shape (1000,)
    :param scale_factor: Scale factor for the x-axis (e.g., 0.3)
    :return: Scaled and interpolated numpy array
    """
    original_length = len(data)
    new_length = int(original_length * scale_factor)
    
    # Create original and new x-coordinates
    x_original = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, new_length)
    
    # Create an interpolation function
    f = interpolate.interp1d(x_original, data, kind='linear')
    
    # Interpolate the data to the new x-coordinates
    scaled_data = f(x_new)
    
    return scaled_data


def time_series_to_str(
    arr, 
    scale=None,               # Scale and interpolate to reduce the text length
    step=None,                # Label every `step` time steps
    csv=False,                # CSV style, position, TODO
    token_per_digit=False,    # Token-per-Digit, llmtime, TODO
    pap=False,                # Prompt-as-Prefix, timellm, TODO
    sep=" "                   # Separator
):
    # Flatten the numpy array
    if type(arr) is list:
        arr = np.array(arr)
    elif type(arr) is not np.ndarray:
        # Torch tensor
        arr = arr.numpy()
        
    flat_arr = arr.flatten()
    
    # Scale the x-axis
    if scale is not None and scale != 1:
        flat_arr = scale_x_axis(flat_arr, scale)

    # Round each element to 2 decimal places
    rounded_arr = np.round(flat_arr, 2)

    # Convert each element to string
    str_arr = [str(i) for i in rounded_arr]

    # Insert time step messages
    if step is not None:
        num_steps = len(str_arr) // step
        for i in range(num_steps + 1):
            index = i * (step + 1)
            # str_arr.insert(index, f'\nstep {i * step} ~ {(i + 1) * step - 1}:')
            str_arr.insert(index, "\n")

    # Join all elements with comma
    result = sep.join(str_arr)

    # Remove comma after colon
    result = result.replace("\n" + sep, "\n")

    # Remove trailing step if there is no comma after last `step`
    if re.search(r"\nstep \d+ ~ \d+:$", result):
        result = re.sub(r", \nstep \d+ ~ \d+:$", "", result)

    return result


def time_series_to_image(
    time_series,
    fig_size=(10, 1.5),
    gt_anomaly_intervals=None,
    anomalies=None
):
    import base64
    from io import BytesIO
    from utils import plot_series_and_predictions, parse_output, interval_to_vector
    import matplotlib.pyplot as plt
    from loguru import logger
    
    if anomalies is not None:
        for method, anomaly in anomalies.items():
            if isinstance(anomaly, str):
                anomaly = parse_output(anomaly)
                anomaly = [[d['start'], d['end']] for d in anomaly]
            if isinstance(anomaly, list) and (len(anomaly) == 0 or len(anomaly[0]) == 2):
                anomaly = interval_to_vector(anomaly, start=0, end=len(time_series))
            anomalies[method] = anomaly
    
    fig = plot_series_and_predictions(
        series=time_series,
        single_series_figsize=fig_size,
        gt_anomaly_intervals=gt_anomaly_intervals,
        anomalies=anomalies
    )
    
    # Encode the figure to a base64 string
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    
    return img_base64
    
        
def create_vision_messages(
    time_series, 
    few_shots=False,
    cot=False,  # TODO
    image_args={}
):
    img = time_series_to_image(time_series, **image_args)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                },
            ],
        }
    ]

    if few_shots:
        history = []
        for series, anom in few_shots:
            img = time_series_to_image(series, **image_args)
            history += [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img}"
                            }
                        }
                    ],
                },
                {"role": "assistant", "content": anom},
            ]
        messages = history + messages
    return messages


def create_text_messages(
    time_series,
    few_shots=False,
    cot=False,  # TODO
    series_args={},
):
    if "scale" not in series_args:
        series_args["scale"] = 1.0
    scale = series_args["scale"]
    
    messages = [
        {
            "role": "user",
            "content": time_series_to_str(time_series, **series_args)
            + "\n\n"
            + LIMIT_PROMPT
            + PROMPT,
        }
    ]

    if few_shots:
        history = []
        for series, anom in few_shots:
            if scale != 1:
                # Scale anom down to the same scale as the time series
                for i in range(len(anom)):
                    anom[i]["start"] = int(anom[i]["start"] * scale)
                    anom[i]["end"] = int(anom[i]["end"] * scale)
            anom = json.dumps(anom)
            
            history += [
                {
                    "role": "user",
                    "content": time_series_to_str(series, **series_args)
                    + "\n\n"
                    + LIMIT_PROMPT
                    + PROMPT,
                },
                {"role": "assistant", "content": anom},
            ]
        messages = history + messages
    return messages


def create_openai_request(
    time_series,
    few_shots=False, 
    vision=False,
    temperature=0.4,
    stop=["’’’’", " – –", "<|endoftext|>", "<|eot_id|>"],
    cot=False,       # Chain of Thought, TODO
    series_args={},  # Arguments for time_series_to_str
    image_args={},   # Arguments for time_series_to_image
):
    if vision:
        messages = create_vision_messages(time_series, few_shots, cot, image_args)
    else:
        messages = create_text_messages(time_series, few_shots, cot, series_args)
    
    return {
        "messages": messages,
        "temperature": temperature,
        "stop": stop
    }
