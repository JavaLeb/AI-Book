package flink.application.study.datastream.operator;

import org.apache.flink.api.common.eventtime.*;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.ProcessJoinFunction;
import org.apache.flink.util.Collector;

import java.time.Duration;

public class IntervalJoinOperator {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        WatermarkStrategy<Integer> watermarkStrategy = new WatermarkStrategy<Integer>() {
            @Override
            public WatermarkGenerator<Integer> createWatermarkGenerator(WatermarkGeneratorSupplier.Context context) {
                return new AscendingTimestampsWatermarks<>();
            }
        }.withTimestampAssigner(new SerializableTimestampAssigner<Integer>() {
            @Override
            public long extractTimestamp(Integer element, long recordTimestamp) {
                return element * 1000;
            }
        });

        DataStream<Integer> leftSource = env.fromData(1, 2, 3, 4, 5, 6)
                .assignTimestampsAndWatermarks(watermarkStrategy);
        DataStream<Integer> rightSource = env.fromData(7, 8, 9, 10)
                .assignTimestampsAndWatermarks(watermarkStrategy);

        KeySelector<Integer,Integer> keySelector = value -> value % 2;

        KeyedStream<Integer, Integer> keyedStream = leftSource.keyBy(keySelector);
        KeyedStream<Integer, Integer> otherKeyedStream = rightSource.keyBy(keySelector);

        DataStream<String> resultDS = keyedStream.intervalJoin(otherKeyedStream)
                .between(Duration.ofMillis(-7000), Duration.ofMillis(7000)) // lower and upper bound
                .upperBoundExclusive() // optional
                .lowerBoundExclusive() // optional
                .process(new ProcessJoinFunction<Integer, Integer, String>() {
                    @Override
                    public void processElement(Integer left, Integer right, ProcessJoinFunction<Integer, Integer, String>.Context ctx, Collector<String> out) throws Exception {
                        out.collect(left + "-" + right);
                    }
                });
        resultDS.print("|-->");

        env.execute();
    }
}
