import org.apache.spark.{SparkConf, SparkContext}

object BroadcastVariables {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val broadcastVar = sc.broadcast(Array(1, 2, 3))

    broadcastVar.value
  }
}
