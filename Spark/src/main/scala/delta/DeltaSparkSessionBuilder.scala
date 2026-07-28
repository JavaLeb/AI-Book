package delta

import org.apache.spark.sql.SparkSession

object DeltaSparkSessionBuilder {

  val spark: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.log.level", "WARN")
    .getOrCreate()

}
