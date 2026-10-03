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
## Week 6 � Final Integration, Testing & 3-Hospital Federated Simulation

Week 6 integration was completed by combining the federated learning server, simulated hospital clients, dataset generation, model training, testing, and simulation workflow.

### Integration Completed

- Integrated the federated learning server and client workflow.
- Extended the simulation from 2 to 3 simulated hospital clients.
- Added Client 2 with a 50/50 class distribution.
- Configured the system for 3 federated clients and 3 training rounds.
- Updated the PowerShell simulation script to launch, monitor, and validate all 3 clients.
- Added client exit-code validation and individual client logs.
- Verified successful FedAvg-based federated training.

### Testing Results

- Pytest: 3/3 tests passed.
- Functional integration tests: 4/4 tests passed.
- Server exit code: 0.
- Client 0 exit code: 0.
- Client 1 exit code: 0.
- Client 2 exit code: 0.

### Federated Training Result

The 3-hospital simulation completed successfully for 3 federated rounds.

| Round | Loss | Accuracy | Latency (sec) |
|---|---:|---:|---:|
| 1 | 0.33089 | 100% | 2.692 |
| 2 | 0.02229 | 100% | 0.381 |
| 3 | 0.00023 | 100% | 0.349 |

The training results above are from the project's synthetic simulation dataset and should not be interpreted as real-world medical model performance.

### Week 6 Status

Final integration, testing, federated simulation, and documentation completed successfully.
