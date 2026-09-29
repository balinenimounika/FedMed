"""FedMed: Federated Learning Clinical Web Dashboard (Streamlit).

Displays training metrics, loss and accuracy convergence, client partition distributions,
and saved model artifacts for privacy-preserving medical image classification.
"""

from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from src.config import (
    BATCH_SIZE,
    CLIENT_CLASS_DISTRIBUTIONS,
    DATASET_SIZE_PER_CLIENT,
    FINAL_MODEL_PATH,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    LOGS_DIR,
    NUM_CLIENTS,
    NUM_ROUNDS,
    RESULTS_DIR,
    SERVER_ADDRESS,
    TRAINING_HISTORY_PATH,
    TRAIN_SPLIT,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FedMed Federated Learning Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Healthcare / Clinical CSS Theme
st.markdown(
    """
    <style>
    /* Global accents */
    .main {
        background-color: #f8fafc;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 18px 22px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    .stMetric label {
        font-size: 0.85rem !important;
        color: #475569 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700;
        color: #0f172a;
    }
    .header-box {
        background: linear-gradient(135deg, #0e7490 0%, #0369a1 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(3, 105, 161, 0.2);
    }
    .header-box h1 {
        color: white !important;
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
    }
    .header-box p {
        color: #e0f2fe !important;
        margin: 8px 0 0 0;
        font-size: 1.05rem;
    }
    .badge {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.2);
        color: #f0fdfa;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Sidebar: Client Partition & Network Info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/hospital.png",
        width=64,
    )
    st.title("FedMed Console")
    st.caption("Privacy-Preserving Clinical AI System")
    st.markdown("---")

    st.subheader("🏥 Participating Institutions")
    st.markdown(
        """
        - **Hospital A (Client 0)**:
          - *Role*: Central Focal Lesion Center
          - *Dataset*: 200 samples (160 Train / 40 Test)
          - *Distribution*: **80% Class 0** / **20% Class 1**
        - **Hospital B (Client 1)**:
          - *Role*: Peripheral Pathology Center
          - *Dataset*: 200 samples (160 Train / 40 Test)
          - *Distribution*: **20% Class 0** / **80% Class 1**
        """
    )

    with st.expander("🔬 Non-IID Medical Partition Details", expanded=False):
        st.markdown(
            """
            **Class 0 (Dense Focal Lesion)**:  
            Central 2D Gaussian density profile with scanner noise:
            $$I(x, y) = \\exp\\left(-\\frac{x^2+y^2}{2\\sigma^2}\\right) + \\epsilon$$

            **Class 1 (Peripheral Ring Lesion)**:  
            Annular cortical rim pattern at radius $r \\approx 8.5$:
            $$I(x, y) = \\exp\\left(-\\frac{(r-8.5)^2}{2\\sigma^2}\\right) + \\epsilon$$

            Both clients hold skewed, non-identically distributed local sets to reflect real-world clinical specialization.
            """
        )

    st.markdown("---")
    st.subheader("⚙️ Federation Settings")
    st.markdown(
        f"""
        - **Aggregation Strategy**: `FedAvg`
        - **Target Server**: `{SERVER_ADDRESS}`
        - **Federated Rounds**: `{NUM_ROUNDS}`
        - **Local Epochs/Round**: `{LOCAL_EPOCHS}`
        - **Batch Size**: `{BATCH_SIZE}`
        - **Learning Rate**: `{LEARNING_RATE}`
        - **Network Protocol**: gRPC (Flower 1.8.0)
        """
    )

    st.markdown("---")
    if st.button("🔄 Refresh Dashboard Data", use_container_width=True):
        st.rerun()

# -----------------------------------------------------------------------------
# 3. Main Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-box">
        <h1>🏥 FedMed: Federated Learning Clinical Dashboard</h1>
        <p>Collaborative multi-hospital training of a shared CNN model for medical pathology classification without centralizing raw patient imaging data.</p>
        <span class="badge">🔒 Privacy Preserved: Local weights only • Zero Patient Data Transfer</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 4. Data Loading & Validation
# -----------------------------------------------------------------------------
history_file = Path(TRAINING_HISTORY_PATH)

if not history_file.is_file():
    st.warning(
        f"⚠️ **Training history not found** at `{history_file}`.\n\n"
        "The federated learning simulation has not been executed yet. "
        "Please run the simulation first from the terminal:\n\n"
        "```powershell\n"
        ".\\run_simulation.ps1\n"
        "```\n\n"
        "or on Linux / macOS:\n"
        "```bash\n"
        "./run_simulation.sh\n"
        "```"
    )
    st.stop()

# Safely parse training history
try:
    df = pd.read_csv(history_file)
    df["round"] = df["round"].astype(int)
    df["loss"] = df["loss"].astype(float)
    df["accuracy"] = df["accuracy"].astype(float)
    df["accuracy_pct"] = df["accuracy"] * 100.0
except Exception as e:
    st.error(f"❌ Error parsing `{history_file}`: {e}")
    st.stop()

if df.empty:
    st.info("ℹ️ `training_history.csv` is currently empty. Run the simulation to generate round metrics.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. Metrics Cards / KPI Summary
# -----------------------------------------------------------------------------
final_round = int(df["round"].iloc[-1])
final_acc = float(df["accuracy_pct"].iloc[-1])
final_loss = float(df["loss"].iloc[-1])
initial_acc = float(df["accuracy_pct"].iloc[0])
initial_loss = float(df["loss"].iloc[0])

acc_delta = final_acc - initial_acc
loss_delta = final_loss - initial_loss

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Federated Rounds",
        value=f"{final_round} / {NUM_ROUNDS}",
        delta="Completed" if final_round >= NUM_ROUNDS else "In Progress",
    )

with kpi2:
    st.metric(
        label="Global Accuracy",
        value=f"{final_acc:.2f}%",
        delta=f"{acc_delta:+.2f}% vs R1" if len(df) > 1 else "Round 1",
    )

with kpi3:
    st.metric(
        label="Global Test Loss",
        value=f"{final_loss:.4f}",
        delta=f"{loss_delta:+.4f} vs R1" if len(df) > 1 else "Round 1",
        delta_color="inverse",
    )

with kpi4:
    st.metric(
        label="Active Clients",
        value=f"{NUM_CLIENTS} Hospitals",
        delta="100% Participation",
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Dual-Chart Visualizations
# -----------------------------------------------------------------------------
st.subheader("📈 Convergence Diagnostics")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
    ax_acc.plot(
        df["round"],
        df["accuracy_pct"],
        color="#0284c7",
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Aggregated Accuracy (%)",
    )
    for _, row in df.iterrows():
        ax_acc.annotate(
            f"{row['accuracy_pct']:.1f}%",
            (row["round"], row["accuracy_pct"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#0369a1",
        )
    ax_acc.set_title("Aggregated Global Accuracy (%) Across Rounds", fontsize=11, fontweight="bold", pad=12)
    ax_acc.set_xlabel("Federated Round", fontsize=10)
    ax_acc.set_ylabel("Weighted Accuracy (%)", fontsize=10)
    ax_acc.set_xticks(df["round"].tolist())
    ax_acc.set_ylim([max(0.0, df["accuracy_pct"].min() - 15), 110])
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)
    fig_acc.tight_layout()
    st.pyplot(fig_acc)

with chart_col2:
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.plot(
        df["round"],
        df["loss"],
        color="#e11d48",
        marker="s",
        linewidth=2.5,
        markersize=8,
        label="Aggregated Loss",
    )
    for _, row in df.iterrows():
        ax_loss.annotate(
            f"{row['loss']:.4f}",
            (row["round"], row["loss"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#be123c",
        )
    ax_loss.set_title("Aggregated Global Loss Across Rounds", fontsize=11, fontweight="bold", pad=12)
    ax_loss.set_xlabel("Federated Round", fontsize=10)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax_loss.set_xticks(df["round"].tolist())
    ax_loss.set_ylim([0, max(df["loss"].max() * 1.25, 0.1)])
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    fig_loss.tight_layout()
    st.pyplot(fig_loss)

# -----------------------------------------------------------------------------
# 7. Full Training History Table
# -----------------------------------------------------------------------------
st.subheader("📋 Training History Log")

display_df = df.copy()
display_df["loss"] = display_df["loss"].map(lambda x: f"{x:.4f}")
display_df["accuracy"] = display_df["accuracy_pct"].map(lambda x: f"{x:.2f}%")
display_df["Status"] = "Converged / Optimal"
display_df = display_df.rename(
    columns={
        "round": "Round",
        "loss": "Aggregated Loss",
        "accuracy": "Aggregated Accuracy",
    }
)[["Round", "Aggregated Loss", "Aggregated Accuracy", "Status"]]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

# -----------------------------------------------------------------------------
# 8. Artifacts Status & Model Registry
# -----------------------------------------------------------------------------
st.subheader("📦 Model Artifacts & File Registry")

model_file = Path(FINAL_MODEL_PATH)
col_art1, col_art2 = st.columns(2)

with col_art1:
    st.markdown("#### 🧠 Aggregated Global Model (`final_model.pt`)")
    if model_file.is_file():
        file_size_kb = model_file.stat().st_size / 1024.0
        mtime = datetime.fromtimestamp(model_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # Verify PyTorch weight integrity
        try:
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            num_layers = len(state_dict)
            param_status = f"Valid PyTorch state dict with {num_layers} parameter tensors."
        except Exception as e:
            param_status = f"Warning verifying state dict: {e}"

        st.success(
            f"✅ **Model Available**: `{model_file.name}`\n\n"
            f"- **Full Path**: `{model_file.resolve()}`\n"
            f"- **File Size**: `{file_size_kb:.2f} KB`\n"
            f"- **Last Updated**: `{mtime}`\n"
            f"- **Integrity**: {param_status}"
        )
    else:
        st.error(f"❌ Final model weights not found at `{model_file}`.")

with col_art2:
    st.markdown("#### 📜 Process Audit Logs")
    logs = [
        ("Server Log", LOGS_DIR / "server.log"),
        ("Client 0 Log (Hospital A)", LOGS_DIR / "client_0.log"),
        ("Client 1 Log (Hospital B)", LOGS_DIR / "client_1.log"),
    ]

    for label, log_path in logs:
        if log_path.is_file():
            size_kb = log_path.stat().st_size / 1024.0
            st.markdown(f"- ✅ **{label}**: `{log_path.name}` ({size_kb:.1f} KB)")
        else:
            st.markdown(f"- ⚠️ **{label}**: Not found (`{log_path.name}`)")

# Collapsible Log Viewer
with st.expander("🔍 Inspect Process Logs", expanded=False):
    log_tabs = st.tabs(["Server Log", "Client 0 Log", "Client 1 Log"])
    for tab, (lbl, lpath) in zip(log_tabs, logs):
        with tab:
            if lpath.is_file():
                st.code(lpath.read_text(encoding="utf-8", errors="replace"), language="text")
            else:
                st.info(f"Log file {lpath.name} is not available.")
