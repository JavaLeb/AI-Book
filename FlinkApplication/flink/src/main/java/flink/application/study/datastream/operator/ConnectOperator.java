package flink.application.study.datastream.operator;

import org.apache.flink.streaming.api.datastream.ConnectedStreams;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.CoFlatMapFunction;
import org.apache.flink.streaming.api.functions.co.CoMapFunction;
import org.apache.flink.util.Collector;

public class ConnectOperator {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(3);

        DataStream<Integer> leftSource = env.fromData(1, 2, 3, 4, 5);
        DataStream<String> rightSource = env.fromData("hadoop", "spark", "flink");

        ConnectedStreams<Integer, String> connectedStreams = leftSource.connect(rightSource);

        DataStream<String> resultDS = connectedStreams.map(new CoMapFunction<Integer, String, String>() {
            @Override
            public String map1(Integer value) throws Exception {
                return value + "|null";
            }

            @Override
            public String map2(String value) throws Exception {
                return "null|" + value;
            }
        });
        resultDS.print("|-->");

        DataStream<String> flatMapDS = connectedStreams.flatMap(new CoFlatMapFunction<Integer, String, String>() {
            @Override
            public void flatMap1(Integer value, Collector<String> out) throws Exception {
                out.collect(value.toString());
            }

            @Override
            public void flatMap2(String value, Collector<String> out) throws Exception {
                out.collect(value);
            }
        });
        flatMapDS.print("|-->");

        env.execute();
    }
}
