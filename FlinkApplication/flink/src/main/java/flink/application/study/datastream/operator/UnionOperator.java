package flink.application.study.datastream.operator;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class UnionOperator {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(3);

        DataStream<Integer> leftSource = env.fromData(1, 2, 3, 4, 5);
        DataStream<Integer> middleSource = env.fromData(1, 7, 8, 9, 10);
        DataStream<Integer> rightSource = env.fromData(1, 12, 13);

        DataStream<Integer> unionDS = leftSource.union(middleSource, rightSource);

        unionDS.print("|-->");

        env.execute();
    }
}
