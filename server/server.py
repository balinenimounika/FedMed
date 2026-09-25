import flwr as fl


def main():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()