"""
Shared schemas, cleaning rules, graph-building logic and lakehouse I/O for the GX notebooks.

Runs unchanged on three platforms:
- Databricks Free Edition (serverless): Unity Catalog tables + a UC volume for files.
  Used by GX_Preprocessing, GX_EDA, GX_MLlib and GX_Streaming.
- Google Colab: local-mode PySpark with the GraphFrames package (not available on serverless).
  Used by GX_GraphFrames.
- Any other machine with PySpark + Java 17 (Linux/macOS/WSL): local mode, parquet "tables".
"""

import os
import shutil

import pyspark
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ
IS_COLAB = "COLAB_RELEASE_TAG" in os.environ or os.path.isdir("/content/drive")

UC_CATALOG, UC_SCHEMA, UC_VOLUME = "workspace", "gx_project", "files"
VOLUME_ROOT = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}"
try:
    _MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # executed as a notebook instead of imported as a module
    _MODULE_DIR = os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))
REPO_DATASETS_DIR = os.path.join(REPO_ROOT, "Datasets")
RAW_FILES = ["mental_health.csv", "twitter_graph.csv"]

if IS_DATABRICKS:
    RAW_DIR, WORK_DIR = f"{VOLUME_ROOT}/raw", VOLUME_ROOT
elif IS_COLAB:
    # Read the CSVs from the repo copy on Google Drive, but keep the (temporary) lakehouse on the
    # Colab VM disk: Spark writes many small files, which is slow on the Drive mount.
    RAW_DIR, WORK_DIR = REPO_DATASETS_DIR, "/content/lakehouse"
else:
    RAW_DIR, WORK_DIR = REPO_DATASETS_DIR, os.path.join(REPO_ROOT, "lakehouse")

SEED = 42
EMOTION_LABELS = ["Anger", "Anxiety", "Depression", "Neutral", "Positive", "Sadness"]
DISTRESS_EMOTIONS = ["Anxiety", "Depression", "Sadness"]

GRAPHFRAMES_VERSION = "0.10.0"  # keep in sync with the `graphframes-py` pip package


def graphframes_package() -> str:
    """Maven coordinates of the GraphFrames JAR matching the installed Spark (3.x/Scala 2.12 or 4.x/Scala 2.13)."""
    if int(pyspark.__version__.split(".")[0]) >= 4:
        return f"io.graphframes:graphframes-spark4_2.13:{GRAPHFRAMES_VERSION}"
    return f"io.graphframes:graphframes-spark3_2.12:{GRAPHFRAMES_VERSION}"


def get_spark(app_name: str) -> SparkSession:
    if IS_DATABRICKS:
        return SparkSession.builder.getOrCreate()
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", os.environ.get("GX_DRIVER_MEMORY", "8g"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.jars.packages", graphframes_package())
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def setup_storage(spark: SparkSession) -> None:
    if IS_DATABRICKS:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}.{UC_VOLUME}")
        _copy_raw_files_to_volume()
    os.makedirs(WORK_DIR, exist_ok=True)


def _copy_raw_files_to_volume() -> None:
    """Land the raw CSVs in the UC volume from the repo's Datasets folder (Databricks Git folder), once."""
    os.makedirs(RAW_DIR, exist_ok=True)
    for filename in RAW_FILES:
        target, source = f"{RAW_DIR}/{filename}", os.path.join(REPO_DATASETS_DIR, filename)
        if os.path.exists(target):
            continue
        if not os.path.exists(source):
            raise FileNotFoundError(f"{filename} is neither in {RAW_DIR} nor in {REPO_DATASETS_DIR}: "
                                    f"upload it to the volume first.")
        shutil.copyfile(source, target)
        print(f"copied {source} -> {target}")


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


def build_graph_silver(bronze_twitter_graph: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Raw relationships -> (silver_edges, silver_vertices).
    Edges: repaired IDs, typed timestamp and a self-interaction flag.
    Vertices: one row per account with first/last activity and its most recent profile snapshot.
    """
    typed_graph = (bronze_twitter_graph
        .withColumn("src", normalize_twitter_id(F.col("Source")))
        .withColumn("dst", normalize_twitter_id(F.col("Destination")))
        .filter(F.col("src").isNotNull() & F.col("dst").isNotNull())
        .withColumn("relationship_ts", F.to_timestamp("Relationship Date (UTC)"))
        .withColumn("src_joined", F.to_timestamp("Source Joined Date"))
        .withColumn("dst_joined", F.to_timestamp("Destination Joined Date")))

    silver_edges = typed_graph.select(
        "src", "dst", F.col("Conversation ID").alias("conversation_id"), "relationship_ts",
        (F.col("src") == F.col("dst")).alias("is_self_interaction"))

    src_side = typed_graph.select(F.col("src").alias("id"), "relationship_ts",
        F.col("Source Followers").alias("followers"), F.col("Source Followed").alias("following"),
        F.col("src_joined").alias("joined_ts"))
    dst_side = typed_graph.select(F.col("dst").alias("id"), "relationship_ts",
        F.col("Destination Followers").alias("followers"), F.col("Destination Followed").alias("following"),
        F.col("dst_joined").alias("joined_ts"))
    observations = src_side.unionByName(dst_side)

    # Profile counts are snapshots taken at relationship time: keep the most recent one per account.
    latest_profile = (observations.filter(F.col("followers").isNotNull())
        .groupBy("id").agg(
            F.max_by("followers", "relationship_ts").alias("followers"),
            F.max_by("following", "relationship_ts").alias("following"),
            F.min("joined_ts").alias("joined_ts")))

    silver_vertices = (observations.groupBy("id")
        .agg(F.min("relationship_ts").alias("first_seen_ts"), F.max("relationship_ts").alias("last_seen_ts"))
        .join(latest_profile, "id", "left")
        .withColumn("has_profile", F.col("followers").isNotNull()))
    return silver_edges, silver_vertices


def build_graph_gold(silver_edges: DataFrame, silver_vertices: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Silver graph -> (gold_graph_edges, gold_account_features).
    Edges: one weighted edge per (src, dst) pair, self-loops removed.
    Accounts: profile + in/out degree, interaction volume, self-replies and account age.
    """
    gold_graph_edges = (silver_edges.filter(~F.col("is_self_interaction"))
        .groupBy("src", "dst")
        .agg(F.count("*").alias("weight"),
             F.countDistinct("conversation_id").alias("conversations"),
             F.min("relationship_ts").alias("first_ts"),
             F.max("relationship_ts").alias("last_ts")))

    out_stats = gold_graph_edges.groupBy(F.col("src").alias("id")).agg(
        F.count("*").alias("out_degree"), F.sum("weight").alias("out_interactions"))
    in_stats = gold_graph_edges.groupBy(F.col("dst").alias("id")).agg(
        F.count("*").alias("in_degree"), F.sum("weight").alias("in_interactions"))
    self_stats = (silver_edges.filter("is_self_interaction")
        .groupBy(F.col("src").alias("id")).agg(F.count("*").alias("self_replies")))

    gold_account_features = (silver_vertices
        .join(out_stats, "id", "left").join(in_stats, "id", "left").join(self_stats, "id", "left")
        .fillna(0, ["out_degree", "out_interactions", "in_degree", "in_interactions", "self_replies"])
        .withColumn("account_age_days", F.datediff("last_seen_ts", "joined_ts")))
    return gold_graph_edges, gold_account_features


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
