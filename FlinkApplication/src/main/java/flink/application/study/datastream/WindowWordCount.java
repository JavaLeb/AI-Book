package flink.application.study.datastream;

import org.apache.flink.api.common.JobExecutionResult;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.execution.JobClient;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.DataStreamSink;
import org.apache.flink.streaming.api.environment.LocalStreamEnvironment;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.util.Collector;

import java.time.Duration;

public class WindowWordCount {

    public static void main(String[] args) throws Exception {
        // 1. 获取执行环境（自动创建）.
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 创建本地执行环境.
//        LocalStreamEnvironment env =StreamExecutionEnvironment.createLocalEnvironment();

        // 本地连接远程集群.
//        StreamExecutionEnvironment env = StreamExecutionEnvironment
//                .createRemoteEnvironment("master", 8081, "target/FlinkApplication-1.0-SNAPSHOT.jar");

        System.out.println("StreamExecutionEnvironment=" + env);
        System.out.println("default parallelism=" + env.getParallelism());

        env.setParallelism(2);
        System.out.println("current parallelism=" + env.getParallelism());

        // 2. 加载/创建初始数据流.
        DataStream<String> initDS = env.socketTextStream("master", 9999);

        // 3. 指定对数据的转换.
        DataStream<Tuple2<String, Integer>> transformedDS = initDS.flatMap(new Splitter())
                .keyBy(value -> value.f0)
                .window(TumblingProcessingTimeWindows.of(Duration.ofSeconds(5)))
                .sum(1);

        // 4. 输出计算结果.
        DataStreamSink<Tuple2<String, Integer>> sinkDS = transformedDS.print();

        // 5. 触发程序执行, 阻塞，等待Job完成.
        JobExecutionResult result = env.execute("WindowWordCount");

//        // 5. 触发程序异步执行.
//        final JobClient jobClient = env.executeAsync("WindowWordCount");
//        System.out.println("JobClient=" + jobClient);
//        // 阻塞，等待Job完成.
//        final JobExecutionResult result = jobClient.getJobExecutionResult().get();

        System.out.println("JobExecutionResult=" + result);

    }

    public static class Splitter implements FlatMapFunction<String, Tuple2<String, Integer>> {
        @Override
        public void flatMap(String sentence, Collector<Tuple2<String, Integer>> out) {
            for (String word : sentence.split(" ")) {
                out.collect(new Tuple2<String, Integer>(word, 1));
            }
        }
    }

}
