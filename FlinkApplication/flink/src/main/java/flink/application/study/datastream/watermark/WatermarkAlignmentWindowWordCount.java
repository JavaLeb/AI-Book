package flink.application.study.datastream.watermark;

import flink.application.study.datastream.Word;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.streaming.api.datastream.*;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

import java.time.Duration;

public class WatermarkAlignmentWindowWordCount {

    public static void main(String[] args) throws Exception {
        // 创建
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1); // 为了尽快触发窗口计算.

        // WatermarkStrategy.
        WatermarkStrategy<String> watermarkStrategy = context -> new MyAscendingTimestampsWatermarks<>();
        watermarkStrategy = watermarkStrategy.withTimestampAssigner((event, timestamp) -> Long.parseLong(event.split(",")[1])) // 提取事件时间戳
                .withWatermarkAlignment("alignment-group", Duration.ofMillis(1000), Duration.ofMillis(200))// 对齐配置
        ;

        SingleOutputStreamOperator<String> slowDS = env.socketTextStream("master", 8888)
                .assignTimestampsAndWatermarks(watermarkStrategy);
        SingleOutputStreamOperator<String> fastDS = env.socketTextStream("master", 9999)
                .assignTimestampsAndWatermarks(watermarkStrategy);



        common(slowDS, watermarkStrategy, "slow");
        common(fastDS, watermarkStrategy, "fast");

        env.execute();

    }

    static void common(DataStream<String> ds, WatermarkStrategy<String> watermarkStrategy, String flag) {
//        SingleOutputStreamOperator<String> watermarkDS = ds.assignTimestampsAndWatermarks(
//                watermarkStrategy);

        SingleOutputStreamOperator<Word> wordDS = ds.map(new Splitter());

        // 按照word分组.
        KeyedStream<Word, String> keyedStream = wordDS.keyBy(new KeySelector<Word, String>() {
            @Override
            public String getKey(Word value) throws Exception {
                return value.getWord();
            }
        });

        // 添加窗口大小为3s的滚动窗口. 注意这里窗口大小的3s也是以事件时间为度量标准，而不是系统时间.
        WindowedStream<Word, String, TimeWindow> windowDS = keyedStream.window(
                TumblingEventTimeWindows.of(Duration.ofMillis(3000))
        );

        // 输出窗口中事件.
        windowDS.apply(new WindowFunction<Word, Word, String, TimeWindow>() {
            public void apply(String s, TimeWindow window, Iterable<Word> input, Collector<Word> out) throws Exception {
                System.out.println(flag + "窗口window=" + window + "，窗口中数据input=" + input);
            }
        });

        // 按照count列统计和.
        windowDS.sum("count")
                .map(word -> flag + "(" + word.getWord() + "," + word.getCount() + ")")
                .print("->");
    }

    public static class Splitter implements MapFunction<String, Word> {
        @Override
        public Word map(String value) throws Exception {
            String[] split = value.split(",");
            return new Word(split[0], 1, Long.valueOf(split[1]));
        }
    }

}
