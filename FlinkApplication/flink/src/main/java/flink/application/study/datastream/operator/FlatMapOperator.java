package flink.application.study.datastream.operator;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

public class FlatMapOperator {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(3);
        // flatMap()
        DataStream<String> source = env.fromData("a flink application", "hadoop application");
        DataStream<String> flatMapDS = source.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public void flatMap(String value, Collector<String> out)
                    throws Exception {
                for (String word : value.split(" ")) {
                    out.collect(word);
                }
            }
        });
        flatMapDS.print("|-->");
        env.execute();
    }
}
