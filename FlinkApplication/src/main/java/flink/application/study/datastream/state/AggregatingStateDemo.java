package flink.application.study.datastream.state;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.state.AggregatingState;
import org.apache.flink.api.common.state.AggregatingStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

public class AggregatingStateDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        DataStreamSource<Long> dataStreamSource = env.fromData(
                3L, 6L, 8L, 3L, 2L, 7L, 1L);

        KeyedStream<Long, Long> keyedStream = dataStreamSource.keyBy(value -> 1L);

        keyedStream
                .flatMap(new AggregatingAvg())
                .print();
        env.execute();
    }

    public static class AggregatingAvg extends RichFlatMapFunction<Long, Double> {

        private transient AggregatingState<Long, Double> aggregatingState;

        @Override
        public void flatMap(Long input, Collector<Double> out) throws Exception {
            aggregatingState.add(input);
            Double aDouble = aggregatingState.get();
            out.collect(aDouble);
        }

        @Override
        public void open(OpenContext openContext) throws Exception {
            // 初始化值状态.
            AggregatingStateDescriptor<Long, Tuple2<Long, Integer>, Double> descriptor =
                    new AggregatingStateDescriptor<>(
                            "descriptor",
                            new MyAggregateFunction(),
                            Types.TUPLE(Types.LONG, Types.INT)
                    );

            this.aggregatingState = getRuntimeContext().getAggregatingState(descriptor);
        }

        class MyAggregateFunction implements AggregateFunction<Long, Tuple2<Long, Integer>, Double> {
            @Override
            public Tuple2<Long, Integer> createAccumulator() {
                return new Tuple2<>(0L, 0);
            }

            @Override
            public Tuple2<Long, Integer> add(Long value, Tuple2<Long, Integer> accumulator) {
                accumulator.f0 += value;
                accumulator.f1 += 1;

                return accumulator;
            }

            @Override
            public Double getResult(Tuple2<Long, Integer> accumulator) {
                return accumulator.f0 * 1.0 / accumulator.f1;
            }

            @Override
            public Tuple2<Long, Integer> merge(Tuple2<Long, Integer> a, Tuple2<Long, Integer> b) {
                return null;
            }
        }
    }

}
