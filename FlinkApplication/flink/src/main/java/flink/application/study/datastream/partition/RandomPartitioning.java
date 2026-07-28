package flink.application.study.datastream.partition;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;

public class RandomPartitioning {
    public static void main(String[] args) throws Exception{
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        DataStream<Integer> source = env.fromData(1, 2, 3, 4, 5, 6);

        DataStream<String> stringDataStream = source.process(new ProcessFunction<Integer, String>() {
            @Override
            public void processElement(Integer value, ProcessFunction<Integer, String>.Context ctx, Collector<String> out) throws Exception {
                int partition = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
                out.collect("value=" + value + " partition=" + partition);
            }
        });
        stringDataStream.print("raw|-->");

        DataStream<Integer> shuffledDS = source.shuffle();
        DataStream<String> resultDS = shuffledDS.process(new ProcessFunction<Integer, String>() {
            @Override
            public void processElement(Integer value, ProcessFunction<Integer, String>.Context ctx, Collector<String> out) throws Exception {
                int partition = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
                out.collect("value=" + value + " partition=" + partition);
            }
        });
        resultDS.print("shuffle|-->");



        env.execute();
    }
}
