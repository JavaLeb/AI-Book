package flink.application.study.datastream.state;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.state.v2.ValueState;
import org.apache.flink.api.common.state.v2.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

public class ValueStateDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        env.fromData(Tuple2.of("s1", 3L),
                        Tuple2.of("s2", 5L),
                        Tuple2.of("s1", 7L),
                        Tuple2.of("s1", 5L),
                        Tuple2.of("s2", 1L))
                .keyBy(value -> value.f0)
                .enableAsyncState() // State V2 API.
                .flatMap(new CountWindowAverage())
                .print("---状态值--->");

        env.execute();
    }

    public static class CountWindowAverage extends RichFlatMapFunction<Tuple2<String, Long>, Tuple4<String, Long, Long, Double>> {

        /**
         * The ValueState handle. 第一个字段计数, 第二个字段求和.
         */
        private transient ValueState<Tuple2<Long, Long>> countSumValueState;

        private Tuple4<String, Long, Long, Double> finalState;

        @Override
        public void flatMap(Tuple2<String, Long> input, Collector<Tuple4<String, Long, Long, Double>> out) throws Exception {
            // access the state value
            countSumValueState.asyncValue().thenApply(hisCountSum -> {
                Tuple2<Long, Long> currentCountSum = (hisCountSum == null ? Tuple2.of(0L, 0L) : hisCountSum);
                currentCountSum.f0 += 1; // 统计元素个数.
                currentCountSum.f1 += input.f1; // 求和.

                return currentCountSum;
            }).thenAccept(currentSum -> {
                countSumValueState.asyncUpdate(currentSum);
                Double avg = 1.0 * currentSum.f1 / currentSum.f0; // 计算平均值.
                finalState = Tuple4.of(input.f0, currentSum.f0, currentSum.f1, avg);
                out.collect(finalState); // 发送计算结果.
            });
        }

        @Override
        public void open(OpenContext ctx) {
            // 初始化状态。
            ValueStateDescriptor<Tuple2<Long, Long>> descriptor =
                    new ValueStateDescriptor<>(
                            "average", // the state name
                            TypeInformation.of(new TypeHint<>() {})); // type information
            countSumValueState = getRuntimeContext().getState(descriptor);
        }

    }

}
