# single device

To setup your experiment please first run

`python download_pretrained_weights.py`

Then you can do whatever offline finetuning you need but the final constraint is that the model you need to submit is that it needs to run on a single A100 or 4090 GPU.


So our evaluation will run

```
docker run --rm -it -p 8085:8085 -p 8082:8082 -v $(pwd)/model_store:/home/model-server/model-store
 -v $(pwd)/config.properties:/home/model-server/config.properties pytorch/torchserve:latest-gpu torchserve --start --model-store model-store --models model_name=model_name.mar --ncs
```

where `model_name.mar` would be a zip file containing your model weights, those weights will be loaded by the torchserve model inferencing framework. You can do this yourself locally to ensure your model performs as you expect and when you're ready to make your submission please upload your `.mar` file to this Azure cloud bucket. 

To package a model you need to run

`torch-model-archiver --model-name model_name --version 1.0 --model-file path_to_model_dot_py --serialized-file path_to_weights.pth`

We will collect
1. Accuracy metrics for a hidden dataset
2. Latency and Throughput metrics

Which you can inspect if you go to `logs/model_log.log`

The winner will be decided based on the performance of those metrics
