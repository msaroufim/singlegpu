# single device

To setup your experiment please first run

`python download_pretrained_weights.py`

Then you can do whatever offline finetuning you need but the final constraint is that the model you need to submit is that it needs to run on a single A100 or 4090 GPU.


So our evaluation will run

```
docker run --rm -it -p 8085:8085 -p 8082:8082 -v $(pwd)/model_store:/home/model-server/model-store
 -v $(pwd)/config.properties:/home/model-server/config.properties pytorch/torchserve:latest-gpu torchserve --start --model-store model-store --models model_name=model_name.mar --ncs
```

This command will actually start a deployed instance of torchserve with some loaded model weights so to evaluate the model we also provide another script `online-evaluate.py`

```python
import requests
import pandas as pd

# Load labeled data from CSV file
labeled_data = pd.read_csv('labeled_data.csv')

# Make a request to the TorchServe instance
url = 'http://<torchserve_host>:<port>/predictions/<model_name>'
payload = {'data': 'your_input_data'}
response = requests.post(url, json=payload)

# Extract the predicted output from the response
predicted_output = response.json()

# Compare the predicted output with labeled data
ground_truth = labeled_data['label'].values[0]  # Assuming label is in the first column
if predicted_output == ground_truth:
    print("Prediction is correct!")
else:
    print("Prediction is incorrect!")
```

where `model_name.mar` would be a zip file containing your model weights, those weights will be loaded by the torchserve model inferencing framework. You can do this yourself locally to ensure your model performs as you expect and when you're ready to make your submission please upload your `.mar` file to this Azure cloud bucket. 

To package a model you need to run

`torch-model-archiver --model-name model_name --version 1.0 --model-file path_to_model_dot_py --serialized-file path_to_weights.pth`

We will collect
1. Accuracy metrics for a hidden dataset
2. Latency and Throughput metrics

Which you can inspect if you go to `logs/model_log.log`

The winner will be decided based on the performance of those metrics
