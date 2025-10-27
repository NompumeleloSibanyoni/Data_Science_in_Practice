from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# --- 1. Initialize Spark session ---
spark = SparkSession.builder.appName("IncomeClassifier").getOrCreate()

# --- 2. Load dataset ---
df = spark.read.csv("data/income.csv", header=True, inferSchema=True)
print("Columns loaded:", df.columns)
df.printSchema()
df.show(5)

# --- 3. Data cleaning ---
df = df.na.drop()
label_column = "income_class"

# --- 4. Identify categorical & numerical columns ---
categorical_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType) and f.name != label_column]
numerical_cols = [f.name for f in df.schema.fields if not isinstance(f.dataType, StringType)]

# --- 5. Encode categorical features ---
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed") for col in categorical_cols + [label_column]]
pipeline = Pipeline(stages=indexers)
df_indexed = pipeline.fit(df).transform(df)

# --- 6. Assemble feature vector ---
indexed_categorical_cols = [f"{col}_indexed" for col in categorical_cols]
feature_cols = indexed_categorical_cols + numerical_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df_indexed)

# --- 7. Prepare final dataframe ---
final_df = df_assembled.select("features", df_assembled[label_column + "_indexed"].alias("label"))

# --- 8. Train/test split ---
train, test = final_df.randomSplit([0.8, 0.2], seed=42)

# --- 9. Train Decision Tree model ---
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxBins=100)
dt_model = dt.fit(train)
dt_predictions = dt_model.transform(test)

# --- 10. Train Random Forest model ---
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxBins=100)
rf_model = rf.fit(train)
rf_predictions = rf_model.transform(test)

# --- 11. Evaluate models ---
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
dt_accuracy = evaluator.evaluate(dt_predictions)
rf_accuracy = evaluator.evaluate(rf_predictions)

print("\nModel Performance Summary:")
print(f"• Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print(f"• Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

spark.stop()
