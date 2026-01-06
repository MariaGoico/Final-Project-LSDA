import time
import sys
import os
import shutil
import numpy as np

from pyspark.sql import SparkSession
from pyspark import StorageLevel
import pyspark.sql.functions as sql_f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

"""
PySpark ML implementation for scalability experiments
"""

BASE_DATA_PATH = "processed_data"  # adjust to your path
BASE_OUTPUT_DIR = "model_outputs"
TARGET_COL = "SNDP"      # target column
IGNORE_COLS = [TARGET_COL, "DATE", "STATION", "NAME", "features", "prediction", "FRSHTT"]
VALID_TYPES = ['int', 'bigint', 'float', 'double', 'tinyint', 'smallint']

MODEL_TYPE = "regression"  # "regression" or "classification"


def get_paths(years):
    return [f"{BASE_DATA_PATH}/climate_{y}.parquet" for y in years]


def evaluate_model(predictions_df, target_col, model_type):
    """
    Evaluate model predictions
    """
    if model_type == "regression":
        evaluator = RegressionEvaluator(
            labelCol=target_col,
            predictionCol="prediction",
            metricName="rmse"
        )
        rmse = evaluator.evaluate(predictions_df)
        return rmse
    else:
        evaluator = MulticlassClassificationEvaluator(
            labelCol=target_col,
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions_df)
        return accuracy


def main(argv):
    # Parse arguments
    cores = int(argv[0])            # number of partitions / Spark cores
    pct = int(argv[1])              # percentage of training data to use
    filename = argv[2] if len(argv) > 2 else "scalability_results.csv"

    MODEL_NAME = f"DecisionTree_cores{cores}_pct{pct}"
    print(f"\n{'='*60}")
    print(f"--- STARTING: {MODEL_NAME} ---")
    print(f"{'='*60}\n")

    # Start Spark
    spark = (
        SparkSession.builder
        .appName(MODEL_NAME)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "12g")
        .config("spark.sql.files.maxPartitionBytes", "128m")
        .config("spark.driver.maxResultSize", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Create output directory
    model_output_dir = os.path.join(BASE_OUTPUT_DIR, MODEL_NAME)
    if os.path.exists(model_output_dir):
        shutil.rmtree(model_output_dir)
    os.makedirs(model_output_dir)

    # Define data splits
    train_years = range(2021, 2023)
    val_years   = range(2023, 2024)

    print("1. Loading Data...")
    try:
        df_train = spark.read.parquet(*get_paths(train_years))
        df_val   = spark.read.parquet(*get_paths(val_years))
    except Exception as e:
        print(f"Error loading data: {e}")
        spark.stop()
        raise e

    # Select features
    dtypes = df_train.dtypes
    feature_cols = [c for c, t in dtypes if t in VALID_TYPES and c not in IGNORE_COLS]
    print(f"   Features ({len(feature_cols)}): {feature_cols[:5]}... (showing first 5)")

    # Prepare training data
    df_train = df_train.select(feature_cols + [TARGET_COL])
    
    # Sample fraction if pct < 100
    if pct < 100:
        print(f"   Sampling {pct}% of total universe...")
        df_train = df_train.sample(withReplacement=False, fraction=pct/100.0, seed=42)


    # Repartition training data
    df_train = df_train.repartition(cores)
    
    # Assemble features
    print("   Assembling features...")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features").setHandleInvalid("skip")
    train_vec = assembler.transform(df_train)
    val_vec = assembler.transform(df_val)

    # Cache and materialize
    train_vec.persist(StorageLevel.MEMORY_AND_DISK)
    train_count = train_vec.count()
    print(f"   Training Rows: {train_count:,}")
    print(f"   Partitions: {train_vec.rdd.getNumPartitions()}")

    # Build model
    print(f"\n2. Training {MODEL_NAME}...")
    if MODEL_TYPE == "regression":
        model = DecisionTreeRegressor(
            featuresCol="features",
            labelCol=TARGET_COL,
            maxDepth=15,
            seed=42
        )
    else:
        model = DecisionTreeClassifier(
            featuresCol="features",
            labelCol=TARGET_COL,
            maxDepth=15,
            seed=42
        )

    start_time = time.time()
    fitted_model = model.fit(train_vec)
    end_time = time.time()

    total_runtime = end_time - start_time
    print(f"   Training completed in {total_runtime:.2f} seconds.")

    # Save model
    print("\n3. Saving Model...")
    fitted_model.write().overwrite().save(os.path.join(model_output_dir, "spark_model"))

    # Evaluate on validation set
    print("\n4. Evaluating on Validation Set...")
    val_preds = fitted_model.transform(val_vec)
    accuracy = evaluate_model(val_preds, TARGET_COL, MODEL_TYPE)
    
    metric_name = "RMSE" if MODEL_TYPE == "regression" else "Accuracy"
    print(f"   Validation {metric_name}: {accuracy:.4f}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Cores: {cores}")
    print(f"  Data fraction: {pct}%")
    print(f"  Training rows: {train_count:,}")
    print(f"  Total runtime: {total_runtime:.2f}s")
    print(f"  Validation {metric_name}: {accuracy:.4f}")
    print(f"{'='*60}\n")

    # Log results to CSV
    with open(filename, "a") as f:
        # Format: cores, pct, total_runtime, train_count, accuracy
        print(f"{cores},{pct},{total_runtime},{train_count},{accuracy}", file=f)

    # Cleanup
    train_vec.unpersist()
    spark.stop()
    
    print("--- PROCESS FINISHED ---\n")


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    main(sys.argv[1:])