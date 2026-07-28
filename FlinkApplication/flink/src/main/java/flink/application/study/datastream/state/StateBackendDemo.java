package flink.application.study.datastream.state;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.StateBackendOptions;


import org.apache.flink.shaded.jackson2.org.yaml.snakeyaml.util.Tuple;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;

public class StateBackendDemo {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        // 查看默认的状态后端.
        String defaultBackend = env.getConfig().toConfiguration().get(StateBackendOptions.STATE_BACKEND);
        System.out.println("默认的状态后端=" + defaultBackend);

        // 设置状态后端.
        Configuration config = new Configuration();
        config.set(StateBackendOptions.STATE_BACKEND, "rocksdb");
        env.configure(config);


        DataStream<String> sourceDS = env.socketTextStream("master", 9999);

        SingleOutputStreamOperator<Tuple<String, Integer>> wordDS = sourceDS.flatMap(new FlatMapFunction<String, Tuple<String, Integer>>() {
            @Override
            public void flatMap(String value, Collector<Tuple<String, Integer>> out) throws Exception {
                String[] split = value.split(" ");
                for (String word : split) {
                    out.collect(new Tuple<>(word, 1));
                }
            }
        });

        // 单词本身作为Key.
        wordDS.keyBy(new KeySelector<Tuple<String, Integer>, String>() {
                    @Override
                    public String getKey(Tuple<String, Integer> tuple) throws Exception {
                        return tuple._1();
                    }
                }).process(new StateProcess())
                .map(v -> "(" + v._1() + "," + v._2() + ")")
                .print("|--->");

        env.execute();
    }

    public static class StateProcess extends ProcessFunction<Tuple<String, Integer>, Tuple<String, Integer>> {
        private transient MapState<String, Integer> mapState;

        @Override
        public void processElement(Tuple<String, Integer> value, ProcessFunction<Tuple<String, Integer>, Tuple<String, Integer>>.Context ctx, Collector<Tuple<String, Integer>> out) throws Exception {
            String word = value._1();
            Integer count = mapState.get(word);
            count = (count == null ? 1 : count + 1);
            mapState.put(word, count);
            out.collect(new Tuple<>(word, count));
        }

        @Override
        public void open(OpenContext openContext) throws Exception {
            super.open(openContext);
            // 初始化值状态.
            MapStateDescriptor<String, Integer> descriptor = new MapStateDescriptor<>(
                    "descriptor",
                    String.class,
                    Integer.class);

            this.mapState = getRuntimeContext().getMapState(descriptor);
        }
    }

}
