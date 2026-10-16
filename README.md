# FedMed: Privacy-Preserving Federated Learning Infrastructure for Medical Image Diagnostics

**Engineering Milestones Documentation: Week 5 & Week 6**  
**Lead AI Infrastructure Architect & Systems Engineer:** Mounika  
**Framework Stack:** Python 3.12 | Flower (`flwr==1.8.0`) | PyTorch (`torch>=2.0.0`) | Streamlit (`streamlit>=1.28.0`)  
**Network Protocol:** gRPC over HTTP/2 | Dual-Platform Automation (PowerShell / POSIX Bash)

---

## 1. Project Overview

### 1.1 Purpose and Objectives
The FedMed framework is an enterprise-grade federated learning (FL) system designed for collaborative clinical machine learning across healthcare institutions. Traditional medical AI requires centralizing diagnostic scans into a single repository, creating severe privacy risks, regulatory liabilities, and data ownership barriers. FedMed addresses this challenge by training diagnostic neural networks collaboratively while keeping patient records, scans, and annotations strictly within local institutional boundaries.

The primary objective of FedMed is to mitigate institutional case-mix bias. Individual medical centers frequently exhibit demographic skews, clinical specializations, and variations in imaging hardware calibrations. By aggregating decentralized model updates across heterogeneous hospital nodes, FedMed builds a globally robust consensus model that achieves superior generalizability compared to models trained at any single institution.

### 1.2 Privacy-Preserving Collaboration
FedMed operates on a zero-raw-data-transfer architecture. Participating hospital nodes train a shared model on local data and transmit only serialized mathematical parameters (weights and biases) to a central aggregation coordinator. The coordinator computes a consolidated update and redistributes the global weights. This decentralized workflow aligns with technical privacy standards under the Health Insurance Portability and Accountability Act (HIPAA, 45 CFR Parts 160/164) and GDPR Article 9.

### 1.3 System Architecture Overview
The system employs a client-server star topology:
- **Central Coordinator (`src/server.py`)**: Manages the federated lifecycle, enforces client quorum barriers, coordinates training rounds, and executes Federated Averaging (FedAvg).
- **Hospital Client Nodes (`src/client.py`)**: Independent client processes representing distinct clinical institutions. Each client executes local training loops over private data partitions and returns updated model weights.
- **Secure Transport Layer**: Model parameters and configuration payloads are serialized and exchanged via gRPC over HTTP/2 with custom payload limits.
- **Monitoring Console (`dashboard.py`)**: An interactive Streamlit dashboard providing real-time convergence tracking, institutional audits, and log inspection.

---

## 2. Server and Client Setup

The Week 5 milestone established the foundational client-server federated learning infrastructure, implementing reliable network transport, local training loops, a convolutional neural network, and model checkpointing.

### 2.1 Flower Server and gRPC Communication
The central coordinator is implemented in `src/server.py` using Flower (`flwr==1.8.0`), listening on `127.0.0.1:8080` via HTTP/2 gRPC. To support multi-layer parameter exchanges without transmission errors, payload limits are explicitly configured to 512 MB:
- `grpc.max_receive_message_length = 536,870,912`
- `grpc.max_send_message_length = 536,870,912`

The server enforces strict quorum policies to prevent partial-round training:
- `fraction_fit = 1.0` and `fraction_evaluate = 1.0`: Requires 100% participation from all registered hospital nodes.
- `min_fit_clients = 3`, `min_evaluate_clients = 3`, `min_available_clients = 3`: Blocks round execution until all three hospital nodes are connected.

### 2.2 Client Implementation and Local Training
Edge client nodes in `src/client.py` implement Flower's `flwr.client.NumPyClient` to decouple PyTorch computation from the networking layer:
- `get_parameters()`: Extracts local PyTorch model weights and converts them into NumPy arrays (`List[np.ndarray]`).
- `fit()`: Receives global weights, updates the local model, trains for $E = 2$ local epochs across private data, and returns updated parameters, sample count, and training latency.
- `evaluate()`: Evaluates the updated global model against an isolated holdout validation set, returning cross-entropy loss and diagnostic accuracy.

Local optimization is performed using mini-batches of size $B = 32$, the Adam optimizer ($\eta = 0.001$, $\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $10^{-4}$), and Cross-Entropy loss.

### 2.3 MedicalCNN Architecture
The diagnostic model (`src/model.py`) is a compact 2D Convolutional Neural Network designed for binary medical classification:

| Layer | Type | Specifications | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| `conv1` | Conv2D + ReLU + MaxPool | In: 1, Out: 16, Kernel: 3x3, Pool: 2x2 | $(B, 16, 14, 14)$ | 160 |
| `conv2` | Conv2D + ReLU + MaxPool | In: 16, Out: 32, Kernel: 3x3, Pool: 2x2 | $(B, 32, 7, 7)$ | 4,640 |
| `fc1` | Linear + ReLU + Dropout | In: 1568, Out: 64, Dropout $p = 0.25$ | $(B, 64)$ | 100,416 |
| `fc2` | Linear (Output Head) | In: 64, Out: 2 (Binary Logits) | $(B, 2)$ | 130 |
| **Total** | **MedicalCNN** | **4 Parameterized Layers** | **Logits $(B, 2)$** | **105,346** |

### 2.4 FedAvg Strategy and Model Checkpointing
The server uses a custom strategy (`FedMedStrategy` / `SaveModelFedAvg`) subclassing `flwr.server.strategy.FedAvg`. In each round, it aggregates parameter updates, calculates sample-weighted validation metrics, and tracks client latencies. Upon completing all rounds, the coordinator automatically saves:
- **Global Checkpoint**: Serialized PyTorch state dictionary saved to `results/final_model.pt` (415 KB).
- **Training History Ledger**: Round-by-round statistics saved to `results/training_history.csv`.

---

## 3.Multi-Node Simulation

The Week 6 milestone scaled the platform into an automated multi-hospital simulation featuring non-IID data partitions, weighted aggregation mathematics, and cross-platform automation scripts.

### 3.1 Three Hospital Clients and Data Distributions
To simulate realistic clinical environments, three distinct institutional profiles were established:
- **Client 0 (Hospital A) – Focal Oncology Center**: Specializes in solid parenchymal tumor detection, with a cohort skewed toward dense focal core nodules (80% Class 0, 20% Class 1).
- **Client 1 (Hospital B) – Cortical Rim Clinic**: Specializes in peripheral inflammatory lesions, with an inverted cohort containing primarily annular rim lesions (20% Class 0, 80% Class 1).
- **Client 2 (Hospital C) – Community General Hospital**: Represents a primary care clinic presenting a balanced patient case mix (50% Class 0, 50% Class 1).

Each institution holds 200 scans partitioned into an 80/20 train/test split (160 training and 40 testing samples), creating a collective multi-center cohort of 600 scans (480 training, 120 testing). Under this non-IID distribution, no single institution possesses sufficient data diversity to train an accurate global model alone.

### 3.2 Synthetic Dataset and Non-IID Partitioning
The dataset (`src/dataset.py`) generates $1 \times 28 \times 28$ normalized grayscale diagnostic images modeling two pathological morphologies:
- **Class 0 (Dense Focal Lesion)**: Simulates solid parenchymal nodules using a 2D Gaussian density function:
  $$I_0(x, y) = A_0 \cdot \exp\left( - \frac{x^2 + y^2}{2\sigma_0^2} \right) + \epsilon(x, y)$$
  with amplitude $A_0 = 0.90$, dispersion $\sigma_0 \in [3.5, 4.1]$, and background noise $\epsilon \sim \mathcal{N}(0, 0.08^2)$.
- **Class 1 (Peripheral Annular Rim)**: Simulates circumferential ring-enhancing lesions:
  $$I_1(x, y) = A_1 \cdot \exp\left( - \frac{(r - r_0)^2}{2\sigma_1^2} \right) + \epsilon(x, y), \quad r = \sqrt{x^2 + y^2}$$
  with amplitude $A_1 = 0.85$, radius $r_0 \in [8.0, 9.0]$, rim thickness $\sigma_1 \in [2.0, 2.4]$, and noise $\epsilon \sim \mathcal{N}(0, 0.08^2)$.

### 3.3 Three-Round Federated Training Workflow
The federated training process executes across three synchronous rounds ($T = 3$):
1. **Global Weight Broadcast**: The server transmits the current global parameter vector $w_t$ to all connected hospital clients.
2. **Local Optimization**: Each client initializes its local `MedicalCNN` with $w_t$ and trains for $E = 2$ epochs using its private training set.
3. **Parameter Upload**: Clients transmit updated weights $w_{t+1}^k$ and sample counts $n_k$ back to the coordinator over gRPC.
4. **Weighted Aggregation**: The server aggregates client updates into unified global weights $w_{t+1}$ using Federated Averaging.
5. **Validation & Logging**: Clients evaluate $w_{t+1}$ on local validation cohorts, reporting loss and accuracy. The server computes sample-weighted global metrics and logs convergence data.

### 3.4 Weighted FedAvg Aggregation Mathematics
The server computes the updated global parameter vector as the sample-weighted linear average of client parameters:

$$w_{t+1} = \sum_{k=0}^{K-1} \frac{n_k}{n} w_{t+1}^k \quad \text{where } n = \sum_{k=0}^{K-1} n_k$$

With equal training cohort sizes ($n_0 = n_1 = n_2 = 160$, total $n = 480$), the formulation simplifies to an equal-weight average:

$$w_{t+1} = \frac{1}{3} w_{t+1}^0 + \frac{1}{3} w_{t+1}^1 + \frac{1}{3} w_{t+1}^2$$

Similarly, global validation accuracy and cross-entropy loss are aggregated using holdout test cohort weights ($n_{\text{test}, k} = 40$, total $n_{\text{test}} = 120$):

$$\text{Accuracy}_{\text{global}}^{(t+1)} = \frac{\sum_{k=0}^{K-1} n_{\text{test}, k} \cdot \text{Accuracy}_k^{(t+1)}}{\sum_{k=0}^{K-1} n_{\text{test}, k}}, \quad \mathcal{L}_{\text{global}}^{(t+1)} = \frac{\sum_{k=0}^{K-1} n_{\text{test}, k} \cdot \mathcal{L}_k^{(t+1)}}{\sum_{k=0}^{K-1} n_{\text{test}, k}}$$

### 3.5 Automation Scripts for Windows and Linux/macOS
FedMed provides dedicated cross-platform runner scripts for reproducible multi-process simulation:
- **Windows PowerShell (`run_simulation.ps1`)**: Launches processes via `ProcessStartInfo` with `cmd.exe /c` wrappers to prevent premature stream closure. It polls TCP port 8080 (30-second timeout) to verify server socket readiness before launching clients, and captures logs into `logs/server.log` and `logs/client_[0-2].log`.
- **Linux/macOS Bash (`run_simulation.sh`)**: Enforces strict execution safety (`set -eo pipefail`), registers signal traps (`trap cleanup SIGINT SIGTERM`) for clean background process termination, and checks port readiness using `nc -z 127.0.0.1 8080`.

**Automated Multi-Process Execution:**
```powershell
# Windows PowerShell
.\run_simulation.ps1
```
```bash
# Linux / macOS Bash
chmod +x run_simulation.sh
./run_simulation.sh
```

**Manual Execution Across Terminals:**
```bash
# Terminal 1: Server Coordinator
python src/server.py --server-address "127.0.0.1:8080" --num-rounds 3

# Terminals 2, 3, 4: Hospital Clients (run in separate terminals)
python src/client.py --client-id 0 --server-address "127.0.0.1:8080"
python src/client.py --client-id 1 --server-address "127.0.0.1:8080"
python src/client.py --client-id 2 --server-address "127.0.0.1:8080"
```

---

## 4. Results and Verification

### 4.1 Multi-Round Convergence Summary
The three-round federated training run demonstrated rapid convergence and steady loss minimization across all three institutions:

| Round | Aggregated Global Loss | Formatted Loss | Global Test Accuracy | Mean Client Latency |
| :---: | :---: | :---: | :---: | :---: |
| **Round 1** | 0.302627 | 0.30263 | 100.0% (1.0000) | 0.842s |
| **Round 2** | 0.012806 | 0.01281 | 100.0% (1.0000) | 0.815s |
| **Round 3** | 0.000070 | 0.00007 | 100.0% (1.0000) | 0.798s |

Global loss decreased from 0.30263 in Round 1 to 0.00007 in Round 3, while client training latency averaged 0.82 seconds per round.

### 4.2 Model Artifacts and Training History
Upon completing the final federated round, the server writes persistent artifacts to the `results/` directory:
- `results/final_model.pt`: Serialized PyTorch state dictionary (415.13 KB) containing all 105,346 parameters of the converged `MedicalCNN`.
- `results/training_history.csv`: Convergence record tracking round index, aggregated loss, test accuracy, and latency.
- `logs/`: Isolated execution logs (`server.log`, `client_0.log`, `client_1.log`, `client_2.log`) documenting communication handshakes and local training progress.

### 4.3 Unit Test Verification
The automated verification suite (`scripts/verify_setup.py`) validates core framework functionality through four unit tests:
1. `test_01_cnn_parameter_serialization`: Verifies lossless parameter extraction and restoration between PyTorch and NumPy arrays.
2. `test_02_dataset_synthesis_and_non_iid_partitioning`: Validates deterministic synthetic scan generation and exact non-IID ratio distribution (80/20, 20/80, 50/50).
3. `test_03_client_fit_and_evaluate`: Tests a complete local training epoch and validation cycle.
4. `test_04_fedavg_aggregation_metric`: Validates sample-weighted parameter averaging and evaluation metric aggregation mathematics.

All 4 unit tests passed successfully in 2.82 seconds. Environment compatibility (Python 3.12, PyTorch 2.14, Flower 1.8.0) is verified by `scripts/check_environment.py`.

### 4.4 Synthetic Benchmark Clarification
The 100.0% validation accuracy and rapid loss convergence reflect the structured nature of the synthetic benchmark dataset. The synthetic morphologies (Gaussian cores vs. annular rings) were specifically engineered with clear geometric separability to formally verify gRPC transport, tensor serialization, and FedAvg mathematics without confounding data quality issues. **This result does not indicate clinical diagnostic readiness on real-world medical imaging**, where noise, subtle margins, and anatomical variability create significantly more complex optimization landscapes.

---

## 5. Dashboard

FedMed provides an interactive clinical web dashboard (`dashboard.py`) built with Streamlit for operational monitoring:

```bash
streamlit run dashboard.py --server.port 8501
```

### Essential Dashboard Components
- **KPI Summary Cards**: Displays key metrics including completed rounds (3/3), global test accuracy (100.0%), final global loss (0.00007) with delta indicators, active clients (3 hospitals), and average training latency.
- **Visual Convergence Curves**: Graphs global accuracy and loss trajectories across rounds. The loss minimization curve is rendered in high-visibility red (`#FF4B4B`).
- **Institutional Profile Audit**: Breaks down cohort distributions across Hospital A, Hospital B, and Hospital C, illustrating local non-IID skews.
- **Model Checkpoint Status**: Displays checkpoint file details (`results/final_model.pt`, 415 KB) and layer structure.
- **Live Execution Log Viewer**: Provides a tabbed interface to inspect runtime logs for the central coordinator and each hospital client node.

---

## 6. Technology Stack and Project Structure

### 6.1 Core Technology Stack

| Technology / Library | Version / Environment | Purpose within FedMed |
| :--- | :--- | :--- |
| **Python** | 3.12 (Virtualenv: `.venv`) | Primary runtime execution environment |
| **Flower (`flwr`)** | `1.8.0` | Federated learning coordination, gRPC transport, and FedAvg strategy |
| **PyTorch (`torch`)** | `>= 2.0.0` | Tensor computation, `MedicalCNN` model, and local optimization |
| **Torchvision** | `>= 0.15.0` | Image transformation utilities |
| **NumPy** | `>= 1.24.0, < 2.0.0` | Synthetic image synthesis and parameter array serialization |
| **Scikit-learn** | `>= 1.3.0` | Metric computation and dataset partitioning |
| **Matplotlib** | `>= 3.7.0` | Telemetry visualization and synthetic sample rendering |
| **Streamlit** | `>= 1.28.0` | Interactive web dashboard and operational monitoring console |

### 6.2 Key Project Files

| File Path | Description |
| :--- | :--- |
| `src/config.py` | Central configuration for hyperparameters, network addresses, and non-IID ratios |
| `src/model.py` | `MedicalCNN` architecture definition, weight getter/setter helpers, and train/test loops |
| `src/dataset.py` | Synthetic medical scan generator and hospital non-IID partitioner |
| `src/client.py` | Flower `NumPyClient` implementation with local training and evaluation hooks |
| `src/server.py` | Flower server coordinator with custom `FedMedStrategy` aggregation strategy |
| `scripts/check_environment.py` | Diagnostic script verifying Python runtime, packages, and directory layout |
| `scripts/verify_setup.py` | Automated unit test suite validating serialization, partitioning, and FedAvg |
| `run_simulation.ps1` | Windows PowerShell multi-process automation runner with port polling |
| `run_simulation.sh` | Linux/macOS POSIX Bash multi-process automation runner with signal cleanup |
| `dashboard.py` | Streamlit clinical monitoring console with KPI cards and log viewer |
| `results/final_model.pt` | Serialized global PyTorch model state dictionary (415 KB) |
| `results/training_history.csv` | Convergence history ledger recording round metrics and latencies |

---

## 7. Limitations and Future Scope

### 7.1 Current System Limitations
1. **Synthetic Data Prototype**: The current implementation operates on synthetic $28 \times 28$ grayscale images. It has not yet been evaluated on clinical imaging modalities such as multi-slice DICOM CT or 3D MRI scans.
2. **Prototype Status**: The software is an academic engineering prototype. It is **not certified as a medical device** under FDA or CE regulations and must not be used for diagnostic decision-making.
3. **Synchronous Network Assumptions**: The simulation assumes persistent client connectivity and does not model real-world network stragglers, high-latency links, or unexpected client dropouts.
4. **Uncertified Compliance**: While data localization adheres to privacy principles, the system has not undergone independent security audits or compliance certification for HIPAA/GDPR.

### 7.2 Future Scope and Roadmap
Planned future development to enhance the FedMed framework includes:
- **Real-World Clinical Datasets**: Integrating public medical benchmarks (e.g., MedMNIST, CheXpert, BraTS) to evaluate performance on high-resolution, multi-class diagnostic tasks.
- **Enhanced Cryptographic Privacy**: Incorporating Differential Privacy (DP) with calibrated Gaussian noise and Homomorphic Encryption (HE via TenSEAL / CKKS) to protect model updates against gradient inversion and membership inference attacks.
- **Asynchronous Aggregation**: Implementing asynchronous federated strategies (e.g., FedAsync, FedProx) to handle stragglers, client dropouts, and heterogeneous compute resources across hospital sites.
- **Transport Security Hardening**: Implementing mutual TLS (mTLS) certificate verification and hardware-backed enclaves (Confidential Computing) for client-server communication channels.
