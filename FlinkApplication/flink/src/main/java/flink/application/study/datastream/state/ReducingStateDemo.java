package flink.application.study.datastream.state;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.state.ReducingState;
import org.apache.flink.api.common.state.ReducingStateDescriptor;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;


public class ReducingStateDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);
        DataStreamSource<Long> dataStreamSource = env.fromData(
                3L, 6L, 8L, 3L, 2L, 7L, 1L);

        KeyedStream<Long, Long> keyedStream = dataStreamSource.keyBy(value -> value % 2);

        keyedStream.flatMap(new ReducingWindowSum()).print("|--->");
        env.execute();
    }

    public static class ReducingWindowSum extends RichFlatMapFunction<Long, Long> {

        private transient ReducingState<Long> reducingState;

        @Override
        public void flatMap(Long input, Collector<Long> out) throws Exception {
            reducingState.add(input); // 直接将输入数据添加到状态中.
            Long state = reducingState.get();
            out.collect(state);
        }

        @Override
        public void open(OpenContext openContext) throws Exception {
            // 初始化值状态.
            ReducingStateDescriptor<Long> descriptor =
                    new ReducingStateDescriptor<>("reducingDescriptor",
                            new ReduceFunction<Long>() {
                                @Override
                                public Long reduce(Long value1, Long value2) throws Exception {
                                    return value1 + value2;
                                }
                            }, Long.class);

            this.reducingState = getRuntimeContext().getReducingState(descriptor);
        }
    }

}
