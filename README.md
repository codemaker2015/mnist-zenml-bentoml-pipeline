# mnist zenml bentoml pipeline

This project is based on top of [MNIST example project](https://github.com/bentoml/gallery/tree/73119b5602b6285678058910fcd53f91a612dccd/pytorch) from BentoML.

- This project is dependent on `python` and runs perfectly on version `3.10+`

- After setting up `python`, install `requirements`
- Next, setup `zenml` by running `zenml init` in the root folder. This creates a `.zen` folder in the root directory that tracks your progress. 
- Inspect the file `zenml_pipeline.py`. Observe the different steps the pipeline is composed of.
- Run the training pipeline that ends up saving the model to registry using
```bash
zenml integration install mlflow
python3 zenml_pipeline.py
```
- Run the generated model via: 
```bash
bentoml serve service.py:MNISTPyTorchService --reload
```
- With the `--reload` flag, the API server will automatically restart when the source file `service.py` is being edited, to boost your development productivity.
- Verify the endpoint can be accessed locally:
```bash
curl -H "Content-Type: multipart/form-data" -F'fileobj=@samples/1.png;type=image/png' http://127.0.0.1:3000/predict_image
```
