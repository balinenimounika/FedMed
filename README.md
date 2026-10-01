# FedMed: Privacy-Preserving Federated Learning Infrastructure for Medical Image Classification

FedMed is a distributed, privacy-preserving Federated Learning (FL) framework engineered for collaborative medical image classification across decentralized healthcare institutions. Built on Flower (`flwr==1.8.0`) and PyTorch (`torch>=2.0.0`), FedMed enables multi-institutional collaborative training of a shared deep Convolutional Neural Network (`MedicalCNN`) without pooling, transmitting, or exposing confidential patient scans.

---

## 1. System Architecture

FedMed implements a synchronous hub-and-spoke federated architecture operating over bidirectional gRPC transport channels. Model parameter exchanges utilize serialized numerical weight vectors (`List[np.ndarray]`), ensuring that patient imaging data strictly resides within institutional boundaries.

### Network Topology & Protocol
- **Transport Layer**: HTTP/2 over gRPC (`127.0.0.1:8080`), configured for high-throughput serialization of PyTorch model parameters.
- **Serialization Standard**: Model layer weights and biases are converted between PyTorch `OrderedDict` state dictionaries and serialized NumPy parameter arrays via `flwr.common.ndarrays_to_parameters` and `flwr.common.parameters_to_ndarrays`.
- **Communication Flow**:
  1. **Global Weight Distribution**: At the start of each round, the central server broadcasts global model parameters $w_t$ to all eligible active clients.
  2. **Decentralized Local Training**: Hospital clients train the model locally across private patient cohorts using mini-batch stochastic gradient descent.
  3. **Parameter Ingestion & Aggregation**: Clients transmit updated parameter vectors $w_{t+1}^k$ and sample volume sizes $n_k$ back to the central server. The server aggregates parameters using Federated Averaging (`FedAvg`).
  4. **Holdout Federated Validation**: The server re-distributes the consolidated global weights $w_{t+1}$ for local evaluation across client validation cohorts, aggregating global loss and sample-weighted accuracy.

### Architectural Node Roles

#### 1. Central Aggregator Server (`src/server.py`)
- Coordinates the overall federated lifecycle across configured training rounds ($T = 3$).
- Executes `FedMedStrategy` (extending `flwr.server.strategy.FedAvg`):
  - Enforces minimum client participation thresholds (`min_fit_clients=2`, `min_evaluate_clients=2`, `min_available_clients=2`).
  - Broadcasts round hyperparameters (e.g., local epoch count $E = 2$).
  - Aggregates parameter updates via sample-weighted linear combination.
  - Computes global sample-weighted evaluation metrics.
- Persists final global model weights to `results/final_model.pt`.
- Logs round-by-round convergence metrics to `results/training_history.csv`.

#### 2. Hospital Clients (`src/client.py`)
- Client 0 (Hospital A) and Client 1 (Hospital B) instantiate `FedMedClient`, subclassing `flwr.client.NumPyClient`.
- Manages local PyTorch `DataLoader` instances with strict private data isolation.
- Implements Flower client hooks:
  - `get_parameters`: Returns current local model parameters to the server.
  - `fit`: Ingests global weights, updates local `MedicalCNN` weights, executes local training epochs using the Adam optimizer, and transmits updated tensors with execution metadata.
  - `evaluate`: Ingests consolidated global weights, evaluates cross-entropy loss and classification accuracy on local holdout test cohorts, and transmits metric tuples.

---

## 2. Non-IID Medical Data Generation & Partitioning

Clinical institutions routinely exhibit distinct patient demographics, case-mix biases, and imaging scanner protocols. FedMed replicates these non-IID (non-Independent and Identically Distributed) data conditions through a synthetic benchmark generating $1 \times 28 \times 28$ single-channel normalized grayscale medical scans.

### Mathematical Formulation of Lesion Classes

Each scan is synthesized on a discrete Cartesian grid centered at the origin:
$$(x, y) \in [-14, 13] \times [-14, 13], \quad r = \sqrt{x^2 + y^2}$$

#### Class 0: Dense Focal Lesion (Central Core Nodule)
Simulates a solid, central parenchymal nodule or tumor core modeled as a two-dimensional Gaussian density distribution:

$$I_{\text{focal}}(x, y) = \exp\left(-\frac{x^2 + y^2}{2\sigma_0^2}\right) + \epsilon(x, y)$$

Where:
- $\sigma_0 \sim \mathcal{U}(3.5, 4.1)$ defines lesion dispersion and spatial volume.
- $\epsilon(x, y) \sim \mathcal{N}(0, 0.08^2)$ simulates background quantum mottle and sensor noise.
- Output intensity values are strictly clamped to the interval $[0.0, 1.0]$.

#### Class 1: Peripheral Annular Rim (Cortical / Wall Lesion)
Simulates a circumferential, ring-enhancing rim lesion situated at the peripheral cortical boundary:

$$I_{\text{rim}}(x, y) = \exp\left(-\frac{(r - r_0)^2}{2\sigma_1^2}\right) + \epsilon(x, y)$$

Where:
- $r_0 \sim \mathcal{U}(8.0, 9.0)$ specifies the mean radial distance of the annular band.
- $\sigma_1 \sim \mathcal{U}(2.0, 2.4)$ dictates the thickness of the cortical rim wall.
- $\epsilon(x, y) \sim \mathcal{N}(0, 0.08^2)$ adds additive Gaussian scanner noise.
- Output intensity values are strictly clamped to the interval $[0.0, 1.0]$.

### Non-IID Institutional Partitioning

Each hospital maintains a local cohort of 200 samples, partitioned into an 80/20 train/test split (160 training samples and 40 testing samples):

- **Client 0 (Hospital A - Focal Oncology Center)**:
  - Total Samples: 200 (160 train / 40 test)
  - Class Distribution: **80% Class 0** (160 samples) and **20% Class 1** (40 samples)
  - Pathological Skew: Heavily skewed toward central focal lesions.

- **Client 1 (Hospital B - Peripheral Pathology Clinic)**:
  - Total Samples: 200 (160 train / 40 test)
  - Class Distribution: **20% Class 0** (40 samples) and **80% Class 1** (160 samples)
  - Pathological Skew: Heavily skewed toward peripheral ring lesions.

- **Federated Cohort Total**:
  - Combined Volume: 400 samples (320 train / 80 test)
  - Combined Distribution: Balanced (200 Class 0 / 200 Class 1)

This non-IID partitioning guarantees that neither hospital can build a generalizable diagnostic classifier in isolation without suffering catastrophic error on the complementary pathological class.

---

## 3. Federated Averaging (FedAvg) Formulation

### Local Client Optimization
In each federated round $t$, the central server distributes global weights $w_t$. Each hospital client $k \in \{0, 1\}$ optimizes its local model parameters over private dataset $D_k$ ($n_k = |D_k| = 160$) for $E = 2$ local epochs using the Adam optimizer with learning rate $\eta = 0.001$ and batch size $B = 32$:

$$w_{t+1}^k \leftarrow \text{LocalTrain}(w_t, D_k)$$

The local objective minimizes the empirical Cross-Entropy loss over mini-batches:

$$\mathcal{L}_k(w) = \frac{1}{|D_k|} \sum_{(x_i, y_i) \in D_k} \ell_{\text{CE}}\left(f(x_i; w), y_i\right)$$

Where $\ell_{\text{CE}}$ represents multi-class cross-entropy on raw logits:

$$\ell_{\text{CE}}(\hat{y}, y) = -\sum_{c=0}^1 y_c \log \left(\frac{\exp(\hat{y}_c)}{\sum_{j=0}^1 \exp(\hat{y}_j)}\right)$$

### Central Server Parameter Aggregation
After local training completes, each client transmits its updated parameter weights $w_{t+1}^k$ to the central server. The server aggregates the distributed parameters via sample-weighted linear averaging:

$$w_{t+1} = \sum_{k=0}^{K-1} \frac{n_k}{n} w_{t+1}^k \quad \text{where } n = \sum_{k=0}^{K-1} n_k$$

Given equal cohort sizes ($n_0 = n_1 = 160, n = 320$), this simplifies to an exact equal-weight average:

$$w_{t+1} = \frac{1}{2} w_{t+1}^0 + \frac{1}{2} w_{t+1}^1$$

### Evaluation Metric Aggregation
Following aggregation, global weights $w_{t+1}$ are evaluated across local holdout validation cohorts $D_{\text{test}, k}$ ($n_{\text{test}, k} = 40$). The server computes sample-weighted global evaluation accuracy and loss:

$$\text{Accuracy}_{\text{global}}^{(t+1)} = \frac{\sum_{k=0}^{K-1} n_{\text{test}, k} \cdot \text{Accuracy}_k^{(t+1)}}{\sum_{k=0}^{K-1} n_{\text{test}, k}}$$

$$\mathcal{L}_{\text{global}}^{(t+1)} = \frac{\sum_{k=0}^{K-1} n_{\text{test}, k} \cdot \mathcal{L}_k^{(t+1)}}{\sum_{k=0}^{K-1} n_{\text{test}, k}}$$

---

## 4. Directory Structure

```text
FedMed/
├── requirements.txt              # Pinned framework and runtime dependencies
├── README.md                     # Comprehensive technical documentation & publication guide
├── dashboard.py                  # Interactive Streamlit clinical web dashboard
├── .gitignore                    # Exclusions for virtual environments, model weights, and logs
├── run_simulation.ps1            # Automated multi-process runner for Windows PowerShell
├── run_simulation.sh             # Automated multi-process runner for Linux/macOS Bash
│
├── src/
│   ├── __init__.py               # Python package initialization marker
│   ├── config.py                 # Central configuration for hyperparameters, network, and paths
│   ├── model.py                  # PyTorch MedicalCNN architecture and train/test helpers
│   ├── dataset.py                # Synthetic medical image generator and non-IID partitioner
│   ├── client.py                 # Flower NumPyClient implementation (FedMedClient)
│   └── server.py                 # Flower Server with custom FedMedStrategy (FedAvg)
│
├── scripts/
│   ├── check_environment.py      # Diagnostic script auditing dependencies and directory layout
│   └── verify_setup.py           # Automated unit test suite (serialization, data, fit, FedAvg)
│
├── results/
│   ├── .gitkeep                  # Preserves directory in version control
│   ├── final_model.pt            # Serialized PyTorch state dictionary of final global model
│   └── training_history.csv      # Round-by-round ledger of aggregated loss and accuracy
│
└── logs/
    ├── .gitkeep                  # Preserves directory in version control
    ├── server.log                # Central Flower gRPC server stdout and stderr
    ├── client_0.log              # Hospital A (Client 0) execution and training log
    └── client_1.log              # Hospital B (Client 1) execution and training log
```

---

## 5. Prerequisites, Environment Setup, & Diagnostics

### System Prerequisites
- **Python**: Version `3.10`, `3.11`, or `3.12` (64-bit)
- **Shell**: PowerShell 5.1+ (Windows) or Bash (Linux/macOS)
- **Hardware**: Standard multi-core x86_64 or ARM CPU (no GPU required)

### Step 1: Virtual Environment Initialization
Clone or navigate to the repository directory:
```bash
cd FedMed
```

Create and activate an isolated virtual environment (`.venv`):

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Pinned Dependencies
Upgrade `pip` and install all required packages from `requirements.txt`:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Core Dependency Inventory:
- `flwr==1.8.0`: Pinned Flower framework ensuring backward-compatible NumPy client bindings.
- `torch>=2.0.0`: Core PyTorch neural network and autograd engine.
- `torchvision>=0.15.0`: Computer vision utilities.
- `numpy>=1.24.0,<2.0.0`: Tensor array manipulation and parameter serialization.
- `scikit-learn>=1.3.0`: Machine learning validation and metric routines.
- `matplotlib>=3.7.0`: Plot generation for clinical diagnostics.
- `streamlit>=1.28.0`: High-performance clinical web dashboard engine.

### Step 3: Run Diagnostic & Verification Suites

#### Diagnostic Environment Audit
Verify the Python runtime version, dependency imports, and required directory structures:
```bash
python scripts/check_environment.py
```

Expected output:
```text
=================================================================
           FedMed Environment & Dependency Diagnostic
=================================================================
  [PASS] Python Version: 3.12.10 (>= 3.10 requirement met)

Package Verification:
-----------------------------------------------------------------
  [PASS] flwr           : 1.8.0        (Required: 1.8.0)
  [PASS] torch          : 2.14.0+cpu   (Required: >=2.0.0)
  [PASS] torchvision    : 0.29.0+cpu   (Required: >=0.15.0)
  [PASS] numpy          : 1.26.4       (Required: >=1.24.0,<2.0.0)
  [PASS] scikit-learn   : 1.9.1        (Required: >=1.3.0)
  [PASS] matplotlib     : 3.11.2       (Required: >=3.7.0)

Directory & File Layout Verification:
-----------------------------------------------------------------
  [PASS] Directory  exists: src
  [PASS] Directory  exists: scripts
  [PASS] Directory  exists: results
  [PASS] Directory  exists: logs
  [PASS] File       exists: src\config.py
  [PASS] File       exists: src\model.py
  [PASS] File       exists: src\dataset.py
  [PASS] File       exists: src\client.py
  [PASS] File       exists: src\server.py
  [PASS] File       exists: requirements.txt
=================================================================
  ALL CHECKS PASSED: FedMed environment is valid and ready.
=================================================================
```

#### Functional Component Unit Tests
Verify parameter serialization, non-IID synthesis, client step execution, and FedAvg math:
```bash
python scripts/verify_setup.py
```

Expected output:
```text
=================================================================
           FedMed Functional Unit Test Suite
=================================================================
test_01_cnn_parameter_serialization: Test CNN parameter extraction and restoration. ... ok
test_02_dataset_synthesis_and_non_iid_partitioning: Test deterministic synthetic generation and 80/20 non-IID skew. ... ok
test_03_client_fit_and_evaluate: Test a complete client fit and evaluate round. ... ok
test_04_fedavg_aggregation_metric: Test FedAvg weighted metric aggregation function. ... ok

----------------------------------------------------------------------
Ran 4 tests in 2.815s

OK
=================================================================
  ALL UNIT TESTS PASSED (4/4 tests successful).
=================================================================
```

---

## 6. Execution Guides

FedMed supports both an **automated multi-process orchestrator** and a **manual multi-terminal workflow**.

### Option A: Automated Multi-Process Execution (Recommended)

The automated script manages process lifecycles, polls port `8080` until the gRPC listener is active, launches clients concurrently, and audits process exit codes.

**Windows (PowerShell):**
```powershell
.\run_simulation.ps1
```

**Linux / macOS (Bash):**
```bash
chmod +x run_simulation.sh
./run_simulation.sh
```

#### Automated Script Flow:
1. Starts `src/server.py` in the background, writing stdout and stderr to `logs/server.log`.
2. Actively polls TCP port `8080` (with a 30-second timeout) until the gRPC socket accepts connections.
3. Spawns `src/client.py --client-id 0` and `src/client.py --client-id 1` in background subprocesses, redirecting output to `logs/client_0.log` and `logs/client_1.log`.
4. Waits synchronously for all 3 federated rounds to complete.
5. Verifies process exit codes and prints `results/training_history.csv`.

---

### Option B: Manual Multi-Terminal Workflow

For low-level inspection or step-by-step debugging, launch the server and clients across separate terminal windows. Ensure `.venv` is activated in each terminal.

#### Terminal 1: Central Server
```bash
python src/server.py --server-address "127.0.0.1:8080" --num-rounds 3
```

#### Terminal 2: Hospital A (Client 0)
```bash
python src/client.py --client-id 0 --server-address "127.0.0.1:8080"
```

#### Terminal 3: Hospital B (Client 1)
```bash
python src/client.py --client-id 1 --server-address "127.0.0.1:8080"
```

---

## 7. Expected Outputs & Benchmark Note

### Execution Results Ledger (`results/training_history.csv`)

During simulation, the central server records round-by-round aggregated metrics to `results/training_history.csv`:

| Round Number | Aggregated Global Loss | Formatted Loss (`.5f`) | Aggregated Accuracy | Convergence State |
| :---: | :---: | :---: | :---: | :---: |
| **Round 1** | `0.30262748152017593` | `0.30263` | `1.0000` (100.00%) | Global Consensus Established |
| **Round 2** | `0.01280635711736977` | `0.01281` | `1.0000` (100.00%) | Optimization Refinement |
| **Round 3** | `0.00006957242658245` | `0.00007` | `1.0000` (100.00%) | Optimal Global Convergence |

### Generated Artifacts
- **`results/final_model.pt`**: Serialized PyTorch state dictionary (415.13 KB) containing the converged weight matrices and bias tensors for `conv1`, `conv2`, `fc1`, and `fc2`.
- **`results/training_history.csv`**: Persistent CSV record of round loss and accuracy.
- **`logs/server.log`**: Detailed Flower server transcript documenting client sampling, round coordination, and gRPC event handling.
- **`logs/client_0.log` & `logs/client_1.log`**: Client execution transcripts documenting local dataset size, training loss progression, and evaluation scores.

---

> [!WARNING]
> ### Mandatory Synthetic Benchmark Disclaimer & Protocol Validation Scope
> The **100.0% validation accuracy** and rapid loss reduction ($0.30263 \rightarrow 0.00007$) observed across federated rounds are an **intentional characteristic of the synthetic benchmark dataset**.
> 
> The benchmark utilizes deterministic, mathematically separable geometric morphology (central Gaussian focal cores vs. peripheral annular rings) designed specifically to:
> 1. Formally verify the federated synchronization protocol.
> 2. Validate weight tensor extraction, serialization, and deserialization routines.
> 3. Verify zero-leakage parameter aggregation across distributed institutional nodes.
> 
> **Clinical Scope Limitation:** In real-world multi-center clinical deployments with high-dimensional, noisy, and heterogeneous pathological imaging (e.g., MedMNIST, ISIC melanoma dermoscopy, CheXpert chest radiographs), models will exhibit non-trivial client drift, significantly lower accuracy ceilings, and complex loss surfaces. This software serves as an architectural prototype for federated infrastructure and is **not intended for clinical diagnostic inference or therapeutic decision-making**.

---

## 8. Interactive Clinical Dashboard (Streamlit)

FedMed features an interactive web dashboard built using Streamlit (`dashboard.py`) to visually communicate training dynamics, convergence metrics, client partition profiles, and artifact integrity.

### Launching the Dashboard

Activate `.venv` and run `dashboard.py` from the project root:

```bash
streamlit run dashboard.py
```

*(On Windows PowerShell, you can also run `.\.venv\Scripts\streamlit.exe run dashboard.py`)*

Access the dashboard in your web browser at:
**`http://localhost:8501`**

### Dashboard Capabilities & Features
- **Clinical Header & Theme**: Medical UI styling with data governance badges highlighting zero patient data transfer.
- **KPI Summary Cards**: Real-time display of:
  - **Federated Rounds**: Completed rounds (`3 / 3`).
  - **Global Accuracy**: `100.00%` with baseline delta (`+0.00% vs R1`).
  - **Global Test Loss**: `0.00007` formatted to 5 decimal places with inverted green delta (`-0.30256 vs R1`).
  - **Active Clients**: `2 Hospitals` with 100% participation.
- **Synthetic Benchmark Disclaimer Banner**: Prominent clinical alert communicating benchmark scope and real-world convergence expectations.
- **Dual Convergence Diagnostics**:
  - **Accuracy Progression Plot**: Matplotlib curve tracking global weighted accuracy across rounds with explicit point annotations.
  - **Loss Minimization Plot**: Matplotlib curve tracking cross-entropy loss reduction across rounds, formatted to 5 decimal places.
- **Full Training History Ledger**: Formatted tabular view of all federated rounds with convergence status indicators.
- **Non-IID Partition Explorer**: Detailed institutional breakdown of Hospital A (80% Class 0 skew) vs. Hospital B (80% Class 1 skew) including mathematical lesion profiles.
- **Model Artifact Registry & Audit Log Viewer**: Verifies `results/final_model.pt` PyTorch state dict integrity and provides tabbed views to inspect raw server and client process logs.

---

## 9. Security, Privacy & Enterprise Hardening

### Data Minimization & Sovereignty
- **Zero Raw Data Transfer**: Raw patient imaging tensors ($x_i, y_i$) never leave the local institutional boundary.
- **Parametric Aggregation**: Only serialized numerical parameter weights ($w_{t+1}^k$) and sample counts ($n_k$) are transmitted across network sockets.
- **Ephemeral In-Memory Buffers**: Client DataLoaders operate strictly in local memory and are released upon process termination.

### Production Enterprise Hardening Roadmap
To transition this prototype into a HIPAA / GDPR-compliant clinical environment, the following security layers should be introduced:
1. **Differential Privacy ($\epsilon, \delta$-DP)**: Implement client-side gradient clipping and Gaussian noise injection (via `Opacus`) to provably defend against model inversion and reconstruction attacks.
2. **Secure Aggregation (SecAgg)**: Integrate cryptographically secure multi-party computation (SMPC) or homomorphic encryption (CKKS) to ensure the central server aggregates model parameters without inspecting individual institutional updates.
3. **mTLS Wire Encryption**: Replace plaintext gRPC with mutual TLS (mTLS) featuring X.509 certificate authentication to secure inter-hospital communications.

---

## 10. Technical Specifications Summary

- **Frameworks**: Flower (`flwr==1.8.0`), PyTorch (`torch>=2.0.0`), Streamlit (`streamlit>=1.28.0`)
- **Model Architecture**: `MedicalCNN`: 2x Conv2D (16, 32 channels) + MaxPool2d + 2x Linear (64, 2 outputs)
- **Input Dimensions**: Grayscale $1 \times 28 \times 28$ normalized tensors ($\text{pixel} \in [0.0, 1.0]$)
- **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss`) on raw unnormalized logits
- **Local Optimizer**: Adam ($\text{learning\_rate} = 0.001$, $\beta_1 = 0.9, \beta_2 = 0.999$)
- **Aggregation Algorithm**: Federated Averaging (`FedAvg`) with sample-weighted metric evaluation
- **Network Protocol**: gRPC (HTTP/2 transport over TCP port `8080`)
- **Federation Topology**: 1 Central Coordinator, 2 Hospital Clients, 3 Sequential Federated Rounds
- **Data Partitioning**: Non-IID Skew: Hospital A (80% Class 0 / 20% Class 1), Hospital B (20% Class 0 / 80% Class 1)
- **Dataset Volume**: 200 samples/client (160 train / 40 test); 400 total across federation
- **Artifact Checkpoint**: `results/final_model.pt` (415.13 KB PyTorch state dictionary)
- **Web Dashboard**: Streamlit application (`dashboard.py`) hosted on `http://localhost:8501`

---

## 11. License

This project is licensed under the MIT License. See `LICENSE` for details.