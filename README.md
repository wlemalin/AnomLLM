<p align="center" >
  <a href="https://github.com/Rose-STL-Lab/AnomLLM"><img src="https://github.com/Rose-STL-Lab/AnomLLM/blob/master/logos/AnomLLM.png?raw=true" width="256" height="256" alt="AnomLLM"></a>
</p>
<h1 align="center">AnomLLM</h1>
<h4 align="center">Can LLMs Understand Time Series Anomalies?
</h4>

<p align="center">
    <a href="https://raw.githubusercontent.com/Rose-STL-Lab/AnomLLM/refs/heads/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="license"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-yellow" alt="python">
    <img src="https://img.shields.io/badge/Version-1.0.0-green" alt="version">
</p>


## | Introduction

We challenge common assumptions about Large Language Models' capabilities in time series understanding. This repository contains the code for reproducing results and **benchmarking** your own large language models' (as long as they are compatible with OpenAI API) anomaly detection capabilities.

## | Citation

[[2410.05440] Can LLMs Understand Time Series Anomalies?](https://arxiv.org/abs/2410.05440)

```
@misc{zhou2024llmsunderstandtimeseries,
      title={Can LLMs Understand Time Series Anomalies?}, 
      author={Zihao Zhou and Rose Yu},
      year={2024},
      eprint={2410.05440},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.05440}, 
}
```

## | Installation

- Dependencies: `conda`
- `PYTHONPATH` shall include `src` directory
- Jpyter notebook path shall be the root directory of the project.

```bash
conda env create --file environment.yml
conda activate anomllm
poetry install
```

## | Dataset Download

We recommend using [`s5cmd`](https://github.com/peak/s5cmd/tree/master) to download the dataset from the NRP S3 bucket.

```bash
s5cmd --no-sign-request --endpoint-url https://s3-west.nrp-nautilus.io cp "s3://anomllm/data/*" data/
```

Alternatively, you can download the dataset from the following link: [Google Drive](https://drive.google.com/file/d/19KNCiOm3UI_JXkzBAWOdqXwM0VH3xOwi/view?usp=sharing) or synthesize your own dataset using `synthesize.sh`. Make sure the dataset is stored in the `data` directory.

## | API Configuration

Create a `credentials.yml` file in the root directory with the following content:

```yaml
gpt-4o:
  api_key: <YOUR_OPENAI_API_KEY>
  base_url: "https://api.openai.com/v1"
gpt-4o-mini:
  api_key: <YOUR_OPENAI_API_KEY>
  base_url: "https://api.openai.com/v1"
gemini-1.5-flash:
  api_key: <YOUR_GOOGLE_API_KEY>
internvlm-76b:
  api_key: <YOUR_LOCAL_OPENAI_SERVER_API_KEY>
  base_url: <YOUR_LOCAL_OPENAI_SERVER_ENDPOINT> (ended with v1)
qwen:
  api_key: <YOUR_LOCAL_OPENAI_SERVER_API_KEY>
  base_url: <YOUR_LOCAL_OPENAI_SERVER_ENDPOINT> (ended with v1)
```

## | Example Usage for Single Time Series

Check out the [example notebook](https://github.com/Rose-STL-Lab/AnomLLM/blob/master/notebook/example.ipynb).

## | Batch Run using OpenAI BatchAPI

`python src/batch_api.py --data $datum --model $model --variant $variant`

See `test.sh` for comprehensive lists of models, variants, and datasets.

## | Online Run using OpenAI API

`python src/online_api.py --data $datum --model $model --variant $variant`

## | License

This project is licensed under the MIT License.