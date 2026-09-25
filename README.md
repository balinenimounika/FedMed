# FedMed – Cross-Silo Federated Learning Engine

FedMed is a privacy-preserving federated learning project for healthcare applications. It is designed to allow multiple hospitals to collaboratively train a machine learning model without directly sharing their private medical data.

## Project Objective

The objective of FedMed is to demonstrate cross-silo federated learning using multiple simulated hospital clients. Each client performs local training and sends model updates to a central server, where the updates are aggregated to create a global model.

## Technology Stack

- Python 3.11
- Flower
- PyTorch
- MONAI
- TenSEAL
- Pytest

## Project Structure

```text
FedMed/
├── server/
├── client/
├── model/
├── data/
├── docs/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore