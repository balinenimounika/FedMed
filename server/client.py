import flwr as fl


class FedMedClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        print("Local hospital training started...")
        return parameters, 1, {}

    def evaluate(self, parameters, config):
        print("Local hospital evaluation started...")
        return 0.0, 1, {"accuracy": 0.0}


def main():
    client = FedMedClient()

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )


if __name__ == "__main__":
    main()