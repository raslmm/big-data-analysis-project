"""Shared schemas, cleaning rules and lakehouse I/O for the GX notebooks (Databricks Free Edition or local PySpark)."""

import os
import shutil

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

UC_CATALOG, UC_SCHEMA, UC_VOLUME = "workspace", "gx_project", "files"
VOLUME_ROOT = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}"
try:
    _MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # executed as a notebook instead of imported as a module
    _MODULE_DIR = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))

RAW_DIR = f"{VOLUME_ROOT}/raw" if IS_DATABRICKS else os.path.join(REPO_ROOT, "Datasets")
WORK_DIR = VOLUME_ROOT if IS_DATABRICKS else os.path.join(REPO_ROOT, "lakehouse")

SEED = 42
EMOTION_LABELS = ["Anger", "Anxiety", "Depression", "Neutral", "Positive", "Sadness"]
DISTRESS_EMOTIONS = ["Anxiety", "Depression", "Sadness"]


def get_spark(app_name: str) -> SparkSession:
    if IS_DATABRICKS:
        return SparkSession.builder.getOrCreate()
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.jars.packages", "io.graphframes:graphframes-spark4_2.13:0.10.0")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def setup_storage(spark: SparkSession) -> None:
    if IS_DATABRICKS:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}.{UC_VOLUME}")
    os.makedirs(WORK_DIR, exist_ok=True)


def raw_path(filename: str) -> str:
    return f"{RAW_DIR}/{filename}"


def work_path(*parts: str) -> str:
    return "/".join([WORK_DIR, *parts])


def reset_path(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def maybe_cache(df: DataFrame) -> DataFrame:
    # Serverless compute does not support cache()/persist().
    return df if IS_DATABRICKS else df.cache()


def table_location(layer: str, name: str) -> str:
    if IS_DATABRICKS:
        return f"{UC_CATALOG}.{UC_SCHEMA}.{layer}_{name}"
    return os.path.join(WORK_DIR, layer, name)


def write_table(df: DataFrame, layer: str, name: str, mode: str = "overwrite") -> str:
    target = table_location(layer, name)
    if IS_DATABRICKS:
        df.write.mode(mode).option("overwriteSchema", "true").saveAsTable(target)
    else:
        df.write.mode(mode).parquet(target)
    print(f"saved {layer}/{name} -> {target}")
    return target


def read_table(spark: SparkSession, layer: str, name: str) -> DataFrame:
    target = table_location(layer, name)
    return spark.read.table(target) if IS_DATABRICKS else spark.read.parquet(target)


def drop_table(spark: SparkSession, layer: str, name: str) -> None:
    if IS_DATABRICKS:
        spark.sql(f"DROP TABLE IF EXISTS {table_location(layer, name)}")
    else:
        reset_path(table_location(layer, name))


MENTAL_HEALTH_SCHEMA = StructType([
    StructField("ID", LongType(), False),
    StructField("Text", StringType(), True),
    StructField("Emotion", StringType(), True),
])

TWITTER_GRAPH_SCHEMA = StructType([
    StructField("Source", StringType(), True),
    StructField("Destination", StringType(), True),
    StructField("Conversation ID", StringType(), True),
    StructField("Relationship Date (UTC)", StringType(), True),
    StructField("Source Joined Date", StringType(), True),
    StructField("Source Followers", DoubleType(), True),
    StructField("Source Followed", DoubleType(), True),
    StructField("Source Verified", StringType(), True),
    StructField("Destination Joined Date", StringType(), True),
    StructField("Destination Followers", DoubleType(), True),
    StructField("Destination Followed", DoubleType(), True),
    StructField("Destination Verified", StringType(), True),
])


def read_mental_health_raw(spark: SparkSession) -> DataFrame:
    return spark.read.csv(raw_path("mental_health.csv"), header=True, schema=MENTAL_HEALTH_SCHEMA,
                          multiLine=True, escape='"')


def read_twitter_graph_raw(spark: SparkSession) -> DataFrame:
    return spark.read.csv(raw_path("twitter_graph.csv"), header=True, schema=TWITTER_GRAPH_SCHEMA)


def normalize_twitter_id(c: Column) -> Column:
    """
    `Source` was exported as a float ('1.60741898676075e+18', '85249550.0') while `Destination`
    keeps the exact integer, so raw strings never match. Both sides are reduced to the same
    15-significant-digit integer key; IDs with <= 15 digits stay exact.
    """
    v = c.cast("double")
    exponent = F.floor(F.log10(v))
    scale = F.pow(F.lit(10.0), exponent - F.lit(14))
    significand = F.round(v / scale).cast("decimal(20,0)")
    long_id = (significand * scale.cast("decimal(20,0)")).cast("decimal(38,0)").cast("string")
    short_id = v.cast("decimal(20,0)").cast("string")
    return F.when(v.isNull() | (v <= 0), None).when(exponent <= 14, short_id).otherwise(long_id)


def is_scientific_notation(c: Column) -> Column:
    return c.rlike(r"[eE][+-]?\d+")


def clean_text(c: Column) -> Column:
    out = F.regexp_replace(c, r"http\S+|www\.\S+", " ")
    out = F.regexp_replace(out, r"@\w+", " ")
    out = F.regexp_replace(out, r"['’`#]", "")
    out = F.regexp_replace(out, r"[^a-zA-Z\s]", " ")
    return F.trim(F.regexp_replace(F.lower(out), r"\s+", " "))


EDA_STOPWORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "feel", "feeling", "im", "like", "get", "dont", "would", "really", "know",
    "one", "even", "much", "also", "ive", "cant", "want",
]

NEGATIONS = {"no", "not", "nor", "never", "don", "dont", "cant", "cannot", "didnt", "doesnt", "isnt",
             "wasnt", "wont", "wouldnt", "shouldnt", "couldnt", "aint", "arent", "havent", "hasnt"}
