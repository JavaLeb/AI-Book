package sql

import org.apache.spark.sql.SparkSession

object SparkSessionDemo {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("SparkSessionDemo")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()
    println("SparkSession=", spark)

    val sc = spark.sparkContext
    println("SparkContext=", sc)

    val sqlContext = spark.sqlContext
    println("SQLContext=", sqlContext)

  }
}
