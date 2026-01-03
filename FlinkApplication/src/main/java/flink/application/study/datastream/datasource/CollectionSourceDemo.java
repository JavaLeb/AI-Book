package flink.application.study.datastream.datasource;


import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.NumberSequenceIterator;

import java.util.Arrays;

public class CollectionSourceDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        DataStreamSource<Integer> ds = env.fromData(1, 2, 3, 4);
        ds.print("|-ds->");

        DataStreamSource<Integer> myTuples = env.fromData(Arrays.asList(1, 2, 3, 4));
        myTuples.print("|-myTuples->");

        DataStreamSource<Long> parallelCollectionDS = env.fromParallelCollection(new NumberSequenceIterator(1, 4), Long.class);
        parallelCollectionDS.print("|-parallelCollectionDS->");

        DataStreamSource<Long> sequenceDS = env.fromSequence(1, 4);
        sequenceDS.print("|-sequenceDS->");

        env.execute();
    }
}
