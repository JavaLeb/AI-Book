package flink.application.study.datastream.state;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.functions.sink.legacy.SinkFunction;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class OperatorStateDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        // 启用检查点.
        env.enableCheckpointing(1000L);

        env.setParallelism(3);

        List<Tuple2<String, Integer>> dataList = Arrays.asList(
                new Tuple2<>("a", 1), new Tuple2<>("b", 2), new Tuple2<>("c", 7),
                new Tuple2<>("a", 3), new Tuple2<>("a", 4),
                new Tuple2<>("b", 6), new Tuple2<>("a", 8));
        DataStreamSource<Tuple2<String, Integer>> sourceDS = env.fromData(dataList);

        KeyedStream<Tuple2<String, Integer>, String> keyedStream = sourceDS.keyBy(
                new KeySelector<Tuple2<String, Integer>, String>() {
                    @Override
                    public String getKey(Tuple2<String, Integer> value) throws Exception {
                        return value.f0;
                    }
                }
        );

        keyedStream.process(new ProcessFunction<Tuple2<String, Integer>, Object>() {
            @Override
            public void processElement(Tuple2<String, Integer> value, ProcessFunction<Tuple2<String, Integer>, Object>.Context ctx, Collector<Object> out) throws Exception {
                int partition = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
                System.out.println("value=" + value + " partition=" + partition);
            }
        });

        keyedStream.addSink(new BufferingSink(3));

        env.execute();
    }


    public static class BufferingSink implements SinkFunction<Tuple2<String, Integer>>, CheckpointedFunction {
        // 缓存的阈值.
        private final int threshold;

        // 状态列表:用于快照时，对缓存进行快照.
        private transient ListState<Tuple2<String, Integer>> checkpointedState;

        // 缓存.
        private List<Tuple2<String, Integer>> bufferedElements;

        public BufferingSink(int threshold) {
            this.threshold = threshold;
            this.bufferedElements = new ArrayList<>();
        }

        @Override
        public void invoke(Tuple2<String, Integer> value, Context context) throws Exception {
            System.out.println(Thread.currentThread().getName() + "-添加元素" + value + "进入缓存");
            bufferedElements.add(value);
            if (bufferedElements.size() >= threshold) {
                for (Tuple2<String, Integer> element : bufferedElements) {
                    // send it to the sink
                    System.out.println(Thread.currentThread().getName() + "-sink:" + element);
                }
                bufferedElements.clear();
            }
        }

        @Override
        public void snapshotState(FunctionSnapshotContext context) throws Exception {
            System.out.println(Thread.currentThread().getName() + "-对缓存" + bufferedElements + "进行快照");
            checkpointedState.update(bufferedElements);
        }

        @Override
        public void initializeState(FunctionInitializationContext context) throws Exception {
            // 初始化 ListState的描述器.
            ListStateDescriptor<Tuple2<String, Integer>> descriptor =
                    new ListStateDescriptor<>("buffered-elements",
                            TypeInformation.of(new TypeHint<Tuple2<String, Integer>>() {
                            }));
            // 初始化ListState.
            checkpointedState = context.getOperatorStateStore().getListState(descriptor);

            if (context.isRestored()) { //恢复后, 将快照中的元素添加到缓存中.
                for (Tuple2<String, Integer> element : checkpointedState.get()) {
                    bufferedElements.add(element);
                }
            }
        }
    }
}
