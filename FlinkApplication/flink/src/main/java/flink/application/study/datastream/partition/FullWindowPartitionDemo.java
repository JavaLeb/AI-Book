package flink.application.study.datastream.partition;

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.datastream.PartitionWindowedStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

public class FullWindowPartitionDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(3);
        DataStream<Integer> dataStream = env.fromData(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        SingleOutputStreamOperator<Integer> mapDS = dataStream.map(new RichMapFunction<Integer, Integer>() {
            @Override
            public Integer map(Integer value) throws Exception {
                int taskIndex = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
                System.out.println("taskIndex=" + taskIndex + ",value=" + value);
                return value;
            }
        });
        PartitionWindowedStream<Integer> partitionWindowedDataStream = mapDS.fullWindowPartition();
        // 计算每个分区的和.
        DataStream<Integer> resultStream = partitionWindowedDataStream.mapPartition(
                new MapPartitionFunction<Integer, Integer>() {
                    @Override
                    public void mapPartition(
                            Iterable<Integer> values, Collector<Integer> out) {
                        int result = 0;
                        for (Integer value : values) {
                            result += value;
                        }
                        out.collect(result);
                    }
                }
        );

        resultStream.print("|--Non-key Stream-->");


        KeyedStream<Integer,Integer> keyedStream =dataStream.keyBy(new KeySelector<Integer, Integer>() {
            @Override
            public Integer getKey(Integer value) throws Exception {
                return value;
            }
        });
        PartitionWindowedStream<Integer> partitionWindowedKeyedDS = keyedStream.fullWindowPartition();
        // 计算每个分区的和.
        DataStream<Integer> resultKeyedStream = partitionWindowedKeyedDS.mapPartition(
                new MapPartitionFunction<Integer, Integer>() {
                    @Override
                    public void mapPartition(
                            Iterable<Integer> values, Collector<Integer> out) {
                        int result = 0;
                        for (Integer value : values) {
                            result += value;
                        }
                        out.collect(result);
                    }
                }
        );

        resultKeyedStream.print("|--Keyed Stream-->");

        env.execute();
    }
}
