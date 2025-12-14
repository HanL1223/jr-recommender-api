"""
Create Model Bundle for Deployment
==================================
Consolidates all training artifacts into a single bundle file for inference.

Usage:
    python scripts/create_model_bundle.py
    python scripts/create_model_bundle.py --artifact-dir models/artifacts
"""

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# -----------------------------------------------------------------------------
# CRITICAL: Add project root to path BEFORE loading pickled models
# This allows pickle to find src.models.* classes
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Bundle Schema
# -----------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """Complete model bundle for deployment."""
    ml_model: Any
    baseline_model: Optional[Any]
    product_features: Any
    customer_profiles: Dict
    prepared_data: Any
    feature_names: list
    encoders: Dict
    cold_start_handler: Optional[Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pickling."""
        return {
            "ml_model": self.ml_model,
            "baseline_model": self.baseline_model,
            "product_features": self.product_features,
            "customer_profiles": self.customer_profiles,
            "prepared_data": self.prepared_data,
            "feature_names": self.feature_names,
            "encoders": self.encoders,
            "cold_start_handler": self.cold_start_handler,
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def load_pickle(path: Path) -> Any:
    """Load a pickle file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> Dict:
    """Load a JSON file with error handling."""
    if not path.exists():
        return {}
    
    with open(path, "r") as f:
        return json.load(f)


def create_cold_start_handler(
    product_features: Any,
    prepared_data: Any,
    customer_profiles: Dict,
) -> Optional[Any]:
    """Create ColdStartHandler instance."""
    try:
        from src.inference.cold_start import ColdStartHandler
        
        handler = ColdStartHandler(
            product_features=product_features,
            prepared_data=prepared_data,
            customer_profiles=customer_profiles,
        )
        logger.info("ColdStartHandler created successfully")
        return handler
    except ImportError as e:
        logger.warning("Could not import ColdStartHandler: %s", e)
        return None
    except Exception as e:
        logger.warning("Could not create ColdStartHandler: %s", e)
        return None


# -----------------------------------------------------------------------------
# Main Bundle Creation
# -----------------------------------------------------------------------------

def create_bundle(
    artifact_dir: Path,
    include_baseline: bool = True,
    include_cold_start: bool = True,
) -> ModelBundle:
    """Create model bundle from training artifacts."""
    logger.info("Loading artifacts from: %s", artifact_dir)
    
    # Load core artifacts
    model_obj = load_pickle(artifact_dir / "recommender.pkl")
    ml_model = model_obj["model"]
    feature_names = model_obj["feature_names"]
    model_params = model_obj.get("params", {})
    
    logger.info("Loaded model: %s", type(ml_model).__name__)
    logger.info("Features: %d columns", len(feature_names))
    
    product_features = load_pickle(artifact_dir / "product_features.pkl")
    logger.info("Loaded product features")
    
    customer_profiles = load_pickle(artifact_dir / "customer_profiles.pkl")
    logger.info("Loaded customer profiles: %d customers", len(customer_profiles))
    
    prepared_data = load_pickle(artifact_dir / "prepared_data.pkl")
    logger.info("Loaded prepared data")
    
    encoders = load_pickle(artifact_dir / "encoders.pkl")
    logger.info("Loaded encoders")
    
    # Load optional artifacts
    model_info = load_json(artifact_dir / "model_info.json")
    if model_info:
        logger.info("Loaded model info (version: %s)", model_info.get("version", "unknown"))
    
    feature_importance = load_json(artifact_dir / "feature_importance.json")
    
    # Create cold start handler
    cold_start_handler = None
    if include_cold_start:
        cold_start_handler = create_cold_start_handler(
            product_features=product_features,
            prepared_data=prepared_data,
            customer_profiles=customer_profiles,
        )
    
    # Load baseline model if exists
    baseline_model = None
    if include_baseline:
        baseline_path = artifact_dir / "baseline_model.pkl"
        if baseline_path.exists():
            baseline_model = load_pickle(baseline_path)
            logger.info("Loaded existing baseline model")
    
    # Build metadata
    metadata = {
        "model_name": model_info.get("model_name", type(ml_model).__name__),
        "model_class": type(ml_model).__name__,
        "version": model_info.get("version", "unknown"),
        "training_date": model_info.get("training_date", "unknown"),
        "params": model_params,
        "valid_metrics": model_info.get("valid_metrics", {}),
        "test_metrics": model_info.get("test_metrics", {}),
        "discovery_metrics": model_info.get("discovery_metrics", {}),
        "discovery_stats": model_info.get("discovery_stats", {}),
        "split_method": model_info.get("split_method", "unknown"),
        "customer_coverage_test": model_info.get("customer_coverage_test", None),
        "n_features": len(feature_names),
        "feature_importance": feature_importance,
        "n_customers": len(customer_profiles),
        "n_products": len(product_features.popularity) if hasattr(product_features, 'popularity') else 0,
        "bundle_created": datetime.now().isoformat(),
        "bundle_version": "v3",
    }
    
    return ModelBundle(
        ml_model=ml_model,
        baseline_model=baseline_model,
        product_features=product_features,
        customer_profiles=customer_profiles,
        prepared_data=prepared_data,
        feature_names=feature_names,
        encoders=encoders,
        cold_start_handler=cold_start_handler,
        metadata=metadata,
    )


def save_bundle(bundle: ModelBundle, output_path: Path) -> None:
    """Save bundle to pickle file."""
    with open(output_path, "wb") as f:
        pickle.dump(bundle.to_dict(), f)
    
    logger.info("Bundle saved to: %s", output_path)
    logger.info("Bundle size: %.2f MB", output_path.stat().st_size / (1024 * 1024))


def print_bundle_summary(bundle: ModelBundle) -> None:
    """Print human-readable bundle summary."""
    print("\n" + "=" * 60)
    print("MODEL BUNDLE SUMMARY")
    print("=" * 60)
    
    meta = bundle.metadata
    
    print("\nModel Information")
    print("-" * 40)
    print(f"  Name:           {meta['model_name']}")
    print(f"  Class:          {meta['model_class']}")
    print(f"  Version:        {meta['version']}")
    print(f"  Training Date:  {meta['training_date']}")
    
    print("\nPerformance Metrics")
    print("-" * 40)
    if meta.get('test_metrics'):
        ndcg3 = meta['test_metrics'].get('ndcg@3', 'N/A')
        if isinstance(ndcg3, (int, float)):
            print(f"  Test NDCG@3:    {ndcg3:.4f}")
    
    if meta.get('discovery_metrics'):
        disc_ndcg = meta['discovery_metrics'].get('ndcg@3')
        if disc_ndcg is not None:
            print(f"  Discovery NDCG: {disc_ndcg:.4f}")
    
    if meta.get('discovery_stats'):
        ds = meta['discovery_stats']
        if ds.get('discovery_hit_rate') is not None:
            print(f"  Discovery Rate: {ds['discovery_hit_rate']:.2%}")
    
    print("\nData Coverage")
    print("-" * 40)
    print(f"  Customers:      {meta['n_customers']:,}")
    print(f"  Products:       {meta['n_products']:,}")
    print(f"  Features:       {meta['n_features']}")
    
    print("\nBundle Components")
    print("-" * 40)
    print(f"  ML Model:       {'Yes' if bundle.ml_model else 'No'}")
    print(f"  Baseline Model: {'Yes' if bundle.baseline_model else 'No'}")
    print(f"  Cold Start:     {'Yes' if bundle.cold_start_handler else 'No'}")
    print(f"  Encoders:       {'Yes' if bundle.encoders else 'No'}")
    
    print("=" * 60 + "\n")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create model bundle for deployment")
    parser.add_argument("--artifact-dir", type=str, default="models/artifacts")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-cold-start", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.exists():
        logger.error("Artifact directory not found: %s", artifact_dir)
        return 1
    
    output_path = Path(args.output) if args.output else artifact_dir / "model_bundle.pkl"
    
    try:
        bundle = create_bundle(
            artifact_dir=artifact_dir,
            include_baseline=not args.no_baseline,
            include_cold_start=not args.no_cold_start,
        )
        
        save_bundle(bundle, output_path)
        
        if not args.quiet:
            print_bundle_summary(bundle)
        
        logger.info("Bundle creation complete")
        return 0
        
    except FileNotFoundError as e:
        logger.error("Missing artifact: %s", e)
        return 1
    except Exception as e:
        logger.error("Bundle creation failed: %s", e)
        raise


if __name__ == "__main__":
    exit(main())