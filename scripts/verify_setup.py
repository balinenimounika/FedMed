"""Unit test suite verifying FedMed core components:
1. CNN parameter serialization & deserialization
2. Synthetic medical dataset generation & non-IID partitioning
3. Client fit and evaluate cycle
4. FedAvg parameter and metric aggregation logic
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_SIZE_PER_CLIENT, IMAGE_DIMS
from src.dataset import generate_client_data, get_client_dataloaders
from src.model import MedicalCNN, get_parameters, set_parameters, test, train
from src.client import FedMedClient
from src.server import evaluate_metrics_aggregation_fn


class TestFedMedSetup(unittest.TestCase):
    """Test suite for FedMed federated learning components."""

    def test_01_cnn_parameter_serialization(self) -> None:
        """Test CNN parameter extraction and restoration."""
        model1 = MedicalCNN(num_classes=2)
        params = get_parameters(model1)
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)

        # Create model2 and assign modified parameters
        model2 = MedicalCNN(num_classes=2)
        modified_params = [p + 0.05 for p in params]
        set_parameters(model2, modified_params)

        params_retrieved = get_parameters(model2)
        for original_mod, retrieved in zip(modified_params, params_retrieved):
            np.testing.assert_allclose(original_mod, retrieved, rtol=1e-5, atol=1e-5)

    def test_02_dataset_synthesis_and_non_iid_partitioning(self) -> None:
        """Test deterministic synthetic generation and 80/20 non-IID skew."""
        # Client 0 test
        c0_imgs, c0_lbls = generate_client_data(client_id=0, total_samples=100, seed=42)
        self.assertEqual(c0_imgs.shape, (100, IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2]))
        self.assertEqual(len(c0_lbls), 100)
        self.assertTrue(np.all((c0_imgs >= 0.0) & (c0_imgs <= 1.0)))

        c0_class0_ratio = (c0_lbls == 0).sum() / 100.0
        self.assertAlmostEqual(c0_class0_ratio, 0.80, delta=0.01)

        # Client 1 test
        c1_imgs, c1_lbls = generate_client_data(client_id=1, total_samples=100, seed=42)
        c1_class1_ratio = (c1_lbls == 1).sum() / 100.0
        self.assertAlmostEqual(c1_class1_ratio, 0.80, delta=0.01)

        # DataLoaders test
        train_loader, test_loader = get_client_dataloaders(client_id=0, batch_size=16)
        train_samples = len(train_loader.dataset)
        test_samples = len(test_loader.dataset)
        self.assertEqual(train_samples + test_samples, DATASET_SIZE_PER_CLIENT)
        self.assertEqual(train_samples, int(DATASET_SIZE_PER_CLIENT * 0.8))
        self.assertEqual(test_samples, int(DATASET_SIZE_PER_CLIENT * 0.2))

    def test_03_client_fit_and_evaluate(self) -> None:
        """Test a complete client fit and evaluate round."""
        train_loader, test_loader = get_client_dataloaders(client_id=0, batch_size=16)
        client = FedMedClient(
            client_id=0,
            train_loader=train_loader,
            test_loader=test_loader,
            device=torch.device("cpu"),
        )

        initial_params = client.get_parameters(config={})
        updated_params, num_samples, fit_metrics = client.fit(
            initial_params, {"server_round": 1, "local_epochs": 1}
        )

        self.assertEqual(num_samples, len(train_loader.dataset))
        self.assertIn("loss", fit_metrics)
        self.assertIn("accuracy", fit_metrics)
        self.assertIsInstance(fit_metrics["loss"], float)
        self.assertIsInstance(fit_metrics["accuracy"], float)

        eval_loss, eval_samples, eval_metrics = client.evaluate(
            updated_params, {"server_round": 1}
        )
        self.assertEqual(eval_samples, len(test_loader.dataset))
        self.assertIn("accuracy", eval_metrics)
        self.assertIsInstance(eval_metrics["accuracy"], float)

    def test_04_fedavg_aggregation_metric(self) -> None:
        """Test FedAvg weighted metric aggregation function."""
        # Client 0: 100 samples, 80% accuracy
        # Client 1: 300 samples, 90% accuracy
        # Expected weighted accuracy: (100*0.80 + 300*0.90) / 400 = (80 + 270) / 400 = 350 / 400 = 0.875
        eval_metrics = [
            (100, {"accuracy": 0.80}),
            (300, {"accuracy": 0.90}),
        ]
        result = evaluate_metrics_aggregation_fn(eval_metrics)
        self.assertIn("accuracy", result)
        self.assertAlmostEqual(result["accuracy"], 0.875, places=5)


def run_tests() -> bool:
    print("=" * 65)
    print("           FedMed Functional Unit Test Suite")
    print("=" * 65)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestFedMedSetup)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 65)
    if result.wasSuccessful():
        print("  ALL UNIT TESTS PASSED (4/4 tests successful).")
        print("=" * 65)
        return True
    else:
        print(f"  UNIT TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors.")
        print("=" * 65)
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
