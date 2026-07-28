package flink.application.study.datastream.partition;

import org.apache.flink.api.common.functions.Partitioner;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;

public class CustomPartitioning {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        Partitioner<Integer> partitioner = new Partitioner<Integer>() {
            @Override
            public int partition(Integer key, int numPartitions) {
                return key % numPartitions;
            }
        };
        KeySelector<Integer, Integer> keySelector = new KeySelector<Integer, Integer>() {
            @Override
            public Integer getKey(Integer value) throws Exception {
                return value;
            }
        };

        DataStream<Integer> source = env.fromData(1, 2, 3, 4, 5, 6);

        DataStream<Integer> partitionedDS = source.partitionCustom(partitioner, keySelector);
        DataStream<String> stringDataStream = partitionedDS.process(new ProcessFunction<Integer, String>() {
            @Override
            public void processElement(Integer value, ProcessFunction<Integer, String>.Context ctx, Collector<String> out) throws Exception {
                int partition = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
                out.collect("value=" + value + " partition=" + partition);
            }
        });

        stringDataStream.print("|-->");

        env.execute();
    }
}
