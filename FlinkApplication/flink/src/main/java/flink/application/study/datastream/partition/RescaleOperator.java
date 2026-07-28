package flink.application.study.datastream.partition;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;

public class RescaleOperator {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        DataStream<Integer> source = env.fromData(1, 2, 3, 4, 5, 6, 7, 8, 9, 10).setParallelism(2);
        source.process(new ProcessFunction<Integer, String>() {
            @Override
            public void processElement(Integer value, ProcessFunction<Integer, String>.Context ctx, Collector<String> out) throws Exception {
                int partition = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
                out.collect("value=" + value + " partition=" + partition);
            }
        }).print("source|-->").setParallelism(1);

        DataStream<Integer> mapDS=source.map(new MapFunction<Integer, Integer>() {
            @Override
            public Integer map(Integer value) throws Exception {
                return value;
            }
        }).setParallelism(6);

        mapDS.process(new ProcessFunction<Integer, String>() {
            @Override
            public void processElement(Integer value, ProcessFunction<Integer, String>.Context ctx, Collector<String> out) throws Exception {
                int partition = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
                out.collect("value=" + value + " partition=" + partition);
            }
        }).print("no rescale|-->").setParallelism(1);

        DataStream<Integer> rescaleDS = source.rescale();

        DataStream<Integer> rescaleMapDS=rescaleDS.map(new MapFunction<Integer, Integer>() {
            @Override
            public Integer map(Integer value) throws Exception {
                return value;
            }
        }).setParallelism(6);

        rescaleMapDS.process(new ProcessFunction<Integer, String>() {
            @Override
            public void processElement(Integer value, ProcessFunction<Integer, String>.Context ctx, Collector<String> out) throws Exception {
                int partition = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
                out.collect("value=" + value + " partition=" + partition);
            }
        }).print("rescale|-->").setParallelism(1);

        env.execute();
    }
}
