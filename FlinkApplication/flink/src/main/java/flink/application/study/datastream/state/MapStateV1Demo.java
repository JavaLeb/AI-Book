package flink.application.study.datastream.state;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

public class MapStateV1Demo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataStreamSource<String> dataStreamSource = env.fromData(
                "spark", "flink", "hadoop", "spark", "hadoop");

        // 单词本身作为Key.
        KeyedStream<String, String> keyedStream = dataStreamSource.keyBy(value -> value);

        keyedStream.flatMap(new MapStateCountV1()).print("|--->");

        env.execute();
    }


    public static class MapStateCountV1 extends RichFlatMapFunction<String, Tuple2<String, Integer>> {

        private transient MapState<String, Integer> mapState;

        @Override
        public void flatMap(String input, Collector<Tuple2<String, Integer>> out) throws Exception {
            Integer count = mapState.get(input);
            count = (count == null ? 1 : count + 1);
            mapState.put(input, count);
            out.collect(new Tuple2<>(input, count));
        }

        @Override
        public void open(OpenContext ctx) {
            // 初始化值状态.
            MapStateDescriptor<String, Integer> descriptor = new MapStateDescriptor<>(
                    "descriptor",
                    String.class,
                    Integer.class);

            this.mapState = getRuntimeContext().getMapState(descriptor);
        }

    }

}
