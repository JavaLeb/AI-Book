package sql

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Encoders}
import sql.SparkSessionTool.spark


object RDD2Dataset {
  def main(args: Array[String]): Unit = {

    // For implicit conversions from RDDs to DataFrames
    import spark.implicits._

    // Create an RDD of Person objects from a text file, convert it to a Dataframe
    val peopleStrRDD: RDD[String] = spark.sparkContext
      .textFile("examples/src/main/resources/people.txt")

    val peopleArrStrRDD: RDD[Array[String]] = peopleStrRDD.map(_.split(","))

    val peopleCaseRDD: RDD[Person] = peopleArrStrRDD
      .map(attributes => Person(attributes(0), attributes(1).trim.toInt))

    val peopleDS:Dataset[Person] = peopleCaseRDD.toDS()

    // Register the DataFrame as a temporary view
    peopleDS.createOrReplaceTempView("people")

    // SQL statements can be run by using the sql methods provided by Spark
    val teenagersDF: DataFrame = spark.sql("SELECT name, age FROM people WHERE age BETWEEN 13 AND 19")

    // The columns of a row in the result can be accessed by field index
    teenagersDF.map(teenager => "Name: " + teenager(0)).show()
    // +------------+
    // |       value|
    // +------------+
    // |Name: Justin|
    // +------------+

    // or by field name
    teenagersDF.map(teenager => "Name: " + teenager.getAs[String]("name")).show()
    // +------------+
    // |       value|
    // +------------+
    // |Name: Justin|
    // +------------+

    // No pre-defined encoders for Dataset[Map[K,V]], define explicitly
    implicit val mapEncoder: Encoder[Map[String, Any]] = Encoders.kryo[Map[String, Any]]
    // Primitive types and case classes can be also defined as
    // implicit val stringIntMapEncoder: Encoder[Map[String, Any]] = ExpressionEncoder()

    // row.getValuesMap[T] retrieves multiple columns at once into a Map[String, T]
    teenagersDF.map(teenager => teenager.getValuesMap[Any](List("name", "age"))).collect()
    // Array(Map("name" -> "Justin", "age" -> 19))
  }
}
