package flink.application.study.datastream.operator;

import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.shaded.jackson2.org.yaml.snakeyaml.util.Tuple;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class KeyByOperator {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(3);

        DataStream<Integer> keyBySource = env.fromData(1, 2, 3, 4, 5);
        KeyedStream<Integer,Integer> keyByDS1=keyBySource.keyBy(new KeySelector<Integer, Integer>() {
            @Override
            public Integer getKey(Integer value) throws Exception {
                return value % 2;
            }
        });

        DataStream<Tuple<Integer, String>> keyBySource2 = env.fromData(
                new Tuple<>(1, "hadoop"),new Tuple<>(1, "flink"),new Tuple<>(2, "spark"));
        KeyedStream<Tuple<Integer, String>,Integer> keyByDS2=keyBySource2.keyBy(new KeySelector<Tuple<Integer, String>, Integer>() {
            @Override
            public Integer getKey(Tuple<Integer, String> value) throws Exception {
                return value._1();
            }
        });

        env.execute();
    }
}
