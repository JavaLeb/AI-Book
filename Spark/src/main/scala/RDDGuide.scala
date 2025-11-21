import org.apache.spark.{SparkConf, SparkContext}

object RDDGuide {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val data = Array(1, 2, 3, 4, 5)
    val distData = sc.parallelize(data)
    distData.foreach(println)
  }
}
