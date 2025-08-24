import bentoml
import mlflow
from model import SimpleConvNet
from train import train, test_model, cross_validate
from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.client import Client
from pydantic import BaseModel

class TrainerConfig(BaseModel):
    """Trainer parameters"""
    epochs: int = 1
    k_folds: int = 2
    lr: float = 0.001


# --- Steps ---
@step
def cross_validate_dataset(config: TrainerConfig) -> dict:
    # Start a new MLflow run for this step/configuration
    with mlflow.start_run(nested=True):
        return cross_validate(
            epochs=config.epochs, k_folds=config.k_folds, learning_rate=config.lr
        )


@step
def train_model(config: TrainerConfig) -> SimpleConvNet:
    # Start a new MLflow run for this step/configuration
    with mlflow.start_run(nested=True):
        return train(epochs=config.epochs, learning_rate=config.lr)


@step
def test_model_performance(model: SimpleConvNet) -> dict:
    return test_model(model=model, _test_loader=None)


@step
def save_model(cv_results: dict, test_results: dict, model: SimpleConvNet) -> None:
    metadata = {
        "acc": float(test_results["correct"]) / test_results["total"],
        "cv_stats": cv_results,
    }

    # ✅ BentoML API for PyTorch
    model_name = "pytorch_mnist"
    bentoml.pytorch.save_model(
        model_name,
        model,
        metadata=metadata,
    )


# --- Pipeline ---
@pipeline(enable_cache=False)
def mnist_pipeline(config: TrainerConfig) -> None:
    """
    Complete MNIST pipeline that handles cross-validation, training, testing, and saving.
    
    Args:
        config: Training configuration parameters
    """
    # Execute steps in sequence with proper artifact flow
    cv_results = cross_validate_dataset(config=config)
    model = train_model(config=config)
    test_results = test_model_performance(model=model)
    save_model(cv_results=cv_results, test_results=test_results, model=model)


if __name__ == "__main__":
    # Run pipeline with different configs
    configs = [
        {"epochs": 1, "k_folds": 2, "lr": 0.0003},
        {"epochs": 2, "k_folds": 2, "lr": 0.0004},
    ]

    for i, config_dict in enumerate(configs):
        print(f"Running pipeline {i+1} with config: {config_dict}")
        
        # Start a new MLflow run for each pipeline execution
        with mlflow.start_run(run_name=f"mnist_pipeline_run_{i+1}"):
            # Create config object
            config = TrainerConfig(**config_dict)
            
            # Log the configuration parameters at pipeline level
            mlflow.log_params({
                "pipeline_epochs": config.epochs,
                "pipeline_k_folds": config.k_folds,
                "pipeline_lr": config.lr
            })
            
            # Run the pipeline with the config
            pipeline_run = mnist_pipeline(config=config)

    # Show MLflow tracking info (if MLflow is in stack)
    client = Client()
    stack = client.active_stack
    if stack.experiment_tracker:
        print(
            f"\nTo inspect experiment runs, start MLflow UI with:\n"
            f"    mlflow ui --backend-store-uri {stack.experiment_tracker.get_tracking_uri()}\n"
        )
    else:
        print("\nNo MLflow experiment tracker configured in the active ZenML stack.\n")