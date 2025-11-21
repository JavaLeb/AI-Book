package sql

import SparkSessionTool.spark
import org.apache.spark.sql.{DataFrame, Dataset}

object DatasetDemo {

  def main(args: Array[String]): Unit = {

    import spark.implicits._

    // Encoders are created for case classes
    val caseClassDS: Dataset[Person] = Seq(Person("Andy", 32)).toDS()
    caseClassDS.show()
    // +----+---+
    // |name|age|
    // +----+---+
    // |Andy| 32|
    // +----+---+

    // Encoders for most common types are automatically provided by importing spark.implicits._
    val primitiveDS: Dataset[Int] = Seq(1, 2, 3).toDS()
    primitiveDS.map(_ + 1).collect() // Returns: Array(2, 3, 4)

    caseClassDS.createOrReplaceTempView("view")

    val result: DataFrame = spark.sql("select name from view")
    result.show()
    // +----+
    // |name|
    // +----+
    // |Andy|
    // +----+

  }
}
