import os

import flwr as fl


def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {"accuracy": 0.0}

    accuracy = sum(
        num_examples * metric["accuracy"]
        for num_examples, metric in metrics
    ) / total_examples

    return {"accuracy": accuracy}


def main():
    num_rounds = int(os.getenv("FEDMED_ROUNDS", "1"))

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()