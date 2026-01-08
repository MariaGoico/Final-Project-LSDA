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

target_col = "log_SNDP"
ignore_cols = [target_col, "DATE", "STATION", "NAME", "features", "prediction", "SNDP", "weight"]
valid_types = ['int', 'bigint', 'float', 'double', 'tinyint', 'smallint']

MODEL_TYPE = "regression"  # "regression" or "classification"

def get_paths(years):
    return [f"{BASE_DATA_PATH}/year={y}" for y in years]


def evaluate_model(predictions_df, target_col, model_type):
    """
    Evaluate model predictions
    """
    evaluator_rmse = RegressionEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="rmse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="r2"
    )
    rmse = evaluator_rmse.evaluate(predictions_df)
    r2 = evaluator_r2.evaluate(predictions_df)
    return rmse, r2



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
    MEM = "5g" 

    spark = SparkSession.builder \
        .appName(MODEL_NAME) \
        .master(f"local[{cores}]") \
        .config("spark.driver.memory", MEM) \
        .config("spark.executor.memory", MEM) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Create output directory
    model_output_dir = os.path.join(BASE_OUTPUT_DIR, MODEL_NAME)
    if os.path.exists(model_output_dir):
        shutil.rmtree(model_output_dir)
    os.makedirs(model_output_dir)

    # Define data splits
    train_years = range(2010, 2023)
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
    feature_cols = [c for c, t in dtypes if t in valid_types and c not in ignore_cols]
    print(f"   Features ({len(feature_cols)}): {feature_cols[:5]}... (showing first 5)")

    # Prepare training data
    df_train = df_train.select(feature_cols + [target_col, "weight"])
    
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
    # train_vec.persist(StorageLevel.MEMORY_AND_DISK)
    train_count = train_vec.count()
    print(f"   Training Rows: {train_count:,}")
    print(f"   Partitions: {train_vec.rdd.getNumPartitions()}")

    # Build model
    print(f"\n2. Training {MODEL_NAME}...")

    model = DecisionTreeRegressor(
        featuresCol="features", 
        labelCol=target_col,
        weightCol="weight",
        maxDepth=5,       # Baseline depth, can be increased (e.g., 10) later
        maxBins=32,       # Number of bins for continuous features
        seed=42
    )

    start_time = time.time()
    fitted_model = model.fit(train_vec)
    end_time = time.time()

    total_runtime = end_time - start_time
    print(f"   Training completed in {total_runtime:.2f} seconds.")

    # Save model
    # print("\n3. Saving Model...")
    # fitted_model.write().overwrite().save(os.path.join(model_output_dir, "spark_model"))

    # Evaluate on validation set
    print("\n4. Evaluating on Validation Set...")
    val_preds = fitted_model.transform(val_vec)
    rmse, r2 = evaluate_model(val_preds, target_col, MODEL_TYPE)

    print(f"   Validation RMSE: {rmse:.4f}")
    print(f"   Validation R2: {r2:.4f}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Cores: {cores}")
    print(f"  Data fraction: {pct}%")
    print(f"  Training rows: {train_count:,}")
    print(f"  Total runtime: {total_runtime:.2f}s")
    print(f"   Validation RMSE: {rmse:.4f}")
    print(f"   Validation R2: {r2:.4f}")
    print(f"{'='*60}\n")

    # Log results to CSV
    with open(filename, "a") as f:
        # Format: cores, pct, total_runtime, train_count, rmse, r2
        print(f"{cores},{pct},{total_runtime},{train_count},{rmse},{r2}", file=f)

    # Cleanup
    train_vec.unpersist()
    spark.stop()
    
    print("--- PROCESS FINISHED ---\n")


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    main(sys.argv[1:])