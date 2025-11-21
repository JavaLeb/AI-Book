package rdd

import org.apache.hadoop.fs.Path
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}

import java.text.SimpleDateFormat
import java.util.Date

object TransformationDemo {
  implicit val intReverseOrdering: Ordering[Int] = Ordering.Int.reverse

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("TransformationDemo")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    sc.setLogLevel("WARN") // 减少日志输出

    val rdd1: RDD[Int] = sc.parallelize(1 to 5, 3) // 2个分区，元素1-10
    val rdd2: RDD[Int] = sc.parallelize(List(1, 2, 2), 3)
    val wordsRdd: RDD[String] = sc.parallelize(List("hello world", "hello spark", "spark scala"), 2)
    val pairRdd: RDD[(String, Int)] = sc.parallelize(List(("a", 1), ("a", 2), ("b", 3), ("b", 4)), 2)
    val rddA: RDD[(String, Int)] = sc.parallelize(List(("a", 1), ("b", 2), ("c", 3)), 2)
    val rddB: RDD[(String, Int)] = sc.parallelize(List(("a", 4), ("b", 5), ("d", 6)), 2)

    val mapRdd: RDD[Int] = rdd1.map(_ * 2)
    println("=== map === 结果: " + mapRdd.collect().mkString(","))

    val filterRdd: RDD[Int] = rdd1.filter(_ % 2 == 0)
    println("=== filter === 结果: " + filterRdd.collect().mkString(","))

    val flatMapRdd: RDD[String] = wordsRdd.flatMap(_.split(" "))
    println("=== flatMap === 结果: " + flatMapRdd.collect().mkString(","))

    val mapPartitionsRdd: RDD[Int] = rdd1.mapPartitions(iterator => iterator.map(_ + 1))
    println("=== mapPartitions === 结果: " + mapPartitionsRdd.collect().mkString(","))

    val mapPartitionsWithIndexRdd: RDD[String] = rdd1.mapPartitionsWithIndex {
      (index, iterator) => iterator.map(x => s"分区$index: $x")
    }
    println("=== mapPartitionsWithIndex === 结果:" + mapPartitionsWithIndexRdd.collect().foreach(println))

    val sampleRdd: RDD[Int] = rdd1.sample(withReplacement = false, fraction = 0.5, seed = 123)
    println("=== sample === 结果: " + sampleRdd.collect().mkString(", "))

    val unionRdd: RDD[Int] = rdd1.union(rdd2)
    println("=== union === 结果: " + unionRdd.collect().mkString(", "))

    val intersectionRdd: RDD[Int] = rdd1.intersection(rdd2)
    println("=== intersection === 结果: " + intersectionRdd.collect().mkString(", "))

    val distinctRdd: RDD[Int] = rdd2.distinct()
    println("=== distinct === 结果: " + distinctRdd.collect().mkString(", "))

    val groupByKeyRdd: RDD[(String, Iterable[Int])] = pairRdd.groupByKey()
    println("=== groupByKey === 结果:")
    groupByKeyRdd.collect().foreach { case (k, v) => println(s"key:$k -> value:${v.mkString(",")}") }

    val reduceByKeyRdd: RDD[(String, Int)] = pairRdd.reduceByKey(_ + _)
    println("=== reduceByKey === 结果: " + reduceByKeyRdd.collect().mkString(","))

    val aggregateByKeyRdd: RDD[(String, Double)] = pairRdd.aggregateByKey((0, 0))(
      (acc, value) => (acc._1 + value, acc._2 + 1), // 分区内聚合
      (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2) // 分区间合并
    ).mapValues { case (sum, count) => sum.toDouble / count }
    println("=== aggregateByKey（平均值） === 结果: " + aggregateByKeyRdd.collect().mkString(", "))

    val sortByKeyRdd: RDD[(String, Int)] = rddA.sortByKey()
    println("=== sortByKey === 结果: " + sortByKeyRdd.collect().mkString(", "))

    val joinRdd: RDD[(String, (Int, Int))] = rddA.join(rddB)
    println("=== join === 结果: " + joinRdd.collect().mkString(", "))

    val cogroupRdd: RDD[(String, (Iterable[Int], Iterable[Int]))] = rddA.cogroup(rddB)
    println("=== cogroup === 结果:")
    cogroupRdd.collect().foreach { case (k, (v1, v2)) =>
      println(s"$k -> (rddA: ${v1.mkString(", ")}, rddB: ${v2.mkString(", ")})")
    }

    val cartesianRdd: RDD[(Int, Int)] = rdd1.cartesian(rdd2)
    println("=== cartesian（前5个元素） === 结果: " + cartesianRdd.collect().mkString(", "))

    // Linux使用grep，Windows使用findstr.
    val pipeRdd: RDD[String] = rdd1.pipe("cmd /c \"findstr 5 || type nul\"")
    println("=== pipe === 结果: " + pipeRdd.collect().mkString(", "))

    val coalesceRdd: RDD[Int] = rdd1.coalesce(2)
    println("=== coalesce（分区数2） === 分区内容:")
    coalesceRdd.glom().collect().foreach(arr => println(arr.mkString(", ")))

    val repartitionRdd: RDD[Int] = rdd1.repartition(5)
    println("=== repartition（分区数5） === 分区内容:")
    repartitionRdd.glom().collect().foreach(arr => println(arr.mkString(", ")))

    val sortedRdd: RDD[(String, Int)] = pairRdd.repartitionAndSortWithinPartitions(new HashPartitioner(2))
    println("=== repartitionAndSortWithinPartitions === 结果:")
    sortedRdd.collect().foreach(println)

    // ====================== Action算子演示 ======================
    val reduceResult: Int = rdd1.reduce(_ + _)
    println(" === reduce === 结果: " + reduceResult)

    val collectResult: Array[Int] = rdd1.collect()
    println("=== collect === 结果: " + collectResult.mkString(", "))

    val countResult: Long = rdd1.count()
    println("=== count === 结果: " + countResult)

    val firstResult: Int = rdd1.first()
    println("=== first === 结果: " + firstResult)


    val takeResult: Array[Int] = rdd1.take(3)
    println("=== take(3) === 结果: " + takeResult.mkString(", "))

    val takeSampleResult: Array[Int] = rdd1.takeSample(withReplacement = true, num = 3, seed = 123)
    println("=== takeSample(withReplacement=true, num=3) === 结果: " + takeSampleResult.mkString(", "))

    val takeOrderedResult: Array[Int] = rdd1.takeOrdered(3) // 升序
    println("=== takeOrdered(3) === 结果: " + takeOrderedResult.mkString(", "))

    val timestamp = new SimpleDateFormat("yyyyMMddHHmmss").format(new Date())
    val uniqueOutputPath = new Path(s"output/text_$timestamp")
    rdd1.saveAsTextFile(uniqueOutputPath.toString)

    val countByKeyResult: collection.Map[String, Long] = pairRdd.countByKey()
    println("=== countByKey === 结果: " + countByKeyResult)

    println("=== foreach === 输出（在executor端打印）:")
    rdd1.foreach(println)

    sc.stop()
  }
}
