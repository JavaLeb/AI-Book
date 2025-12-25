package flink.application.study.datastream.state;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.configuration.*;
import org.apache.flink.core.execution.CheckpointingMode;
import org.apache.flink.shaded.jackson2.org.yaml.snakeyaml.util.Tuple;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.CheckpointConfig;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;
import org.apache.flink.util.ParameterTool;
import org.apache.flink.util.StringUtils;

public class CheckpointDemo {
    public static void main(String[] args) throws Exception {
        // 从main()方法中解析参数.
        ParameterTool parameters = ParameterTool.fromArgs(args);

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        // 获取检查点默认配置,并打印.
        CheckpointConfig cpConf = env.getCheckpointConfig();
        System.out.println("Checkpoint Enable默认值=" + cpConf.isCheckpointingEnabled());
        System.out.println("Checkpoint Mode默认值=" + cpConf.getCheckpointingConsistencyMode());
        System.out.println("Checkpoint Timeout默认值=" + cpConf.getCheckpointTimeout());
        System.out.println("Min Time Between Checkpoints默认值=" + cpConf.getMinPauseBetweenCheckpoints());
        System.out.println("Max Concurrent Checkpoints默认值=" + cpConf.getMaxConcurrentCheckpoints());
        System.out.println("Tolerable Checkpoint Failure Number默认值=" + cpConf.getTolerableCheckpointFailureNumber());

        // 其它默认检查点配置.
        Configuration config = cpConf.toConfiguration();
        System.out.println("其他默认配置=" + config);
        // 输出null, 当检查点存储为null时，Flink直接初始化为JobManagerCheckpointStorage.
        System.out.println(config.get(CheckpointingOptions.CHECKPOINT_STORAGE));


        // 启动检查点, 并配置检查点模式.
        env.enableCheckpointing(10 * 1000, CheckpointingMode.EXACTLY_ONCE);
        // 检查点存储.
        config.set(CheckpointingOptions.CHECKPOINT_STORAGE, "filesystem");
        // 文件系统类型:file或hdfs.
        String fs = parameters.get("checkpoint.storage.fs");
        if (StringUtils.isNullOrWhitespaceOnly(fs)) {
            fs = "file";
        }
        config.set(CheckpointingOptions.CHECKPOINTS_DIRECTORY, fs + ":///flink-checkpoints-data/");
        // 检查点存储类型为filesystem时, 写缓存大小.
        config.set(CheckpointingOptions.FS_WRITE_BUFFER_SIZE, 16);
        // 文件最小大小.
        config.set(CheckpointingOptions.FS_SMALL_FILE_THRESHOLD, new MemorySize(1024));
        // 最大保留检查点数.
        config.set(CheckpointingOptions.MAX_RETAINED_CHECKPOINTS, 2);
        // Job取消时, 是否保留检查点.
        cpConf.setExternalizedCheckpointRetention(ExternalizedCheckpointRetention.RETAIN_ON_CANCELLATION);

        // 用于本地IDEA运行时，状态恢复.
        String checkpoint = parameters.get("execution.savepoint.path");
        if (!StringUtils.isNullOrWhitespaceOnly(checkpoint)) {
            System.out.println("检查点恢复路径=" + checkpoint);
            config.setString("execution.savepoint.path", checkpoint);
        }
        // Job完成是否继续启动检查点.
        config.set(CheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, false);
        env.configure(config);

        System.out.println("最终外部检查点保留策略=" + cpConf.getExternalizedCheckpointRetention());
        System.out.println("最终配置信息：\n" + env.getConfig());


        DataStream<String> sourceDS = env.socketTextStream("master", 9999);

        // 单行中的单词，以空格分隔.
        SingleOutputStreamOperator<Tuple<String, Integer>> wordDS = sourceDS.map(new MapFunction<String, Tuple<String, Integer>>() {
            @Override
            public Tuple<String, Integer> map(String value) throws Exception {
                String[] split = value.split(" ");
                return new Tuple<>(split[0], 1);
            }
        });

        // 单词本身作为Key.
        KeyedStream<Tuple<String, Integer>, String> keyedStream = wordDS.keyBy(new KeySelector<Tuple<String, Integer>, String>() {
            @Override
            public String getKey(Tuple<String, Integer> tuple) throws Exception {
                return tuple._1();
            }
        });

        keyedStream.flatMap(new StateCount()).print("|--->");

        env.execute();

    }

    public static class StateCount extends RichFlatMapFunction<Tuple<String, Integer>, String> {

        private transient MapState<String, Integer> mapState;

        @Override
        public void flatMap(Tuple<String, Integer> tuple, Collector<String> out) throws Exception {
            String word = tuple._1();
            Integer count = mapState.get(word);
            count = (count == null ? 1 : count + 1);
            mapState.put(word, count);
            out.collect("(" + word + "," + count + ")");
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
