package flink.application.study.datastream.watermark;

import flink.application.study.datastream.Word;
import org.apache.flink.api.common.eventtime.BoundedOutOfOrdernessWatermarks;
import org.apache.flink.api.common.eventtime.SerializableTimestampAssigner;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.datastream.WindowedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

import java.time.Duration;

public class BOWatermarkWindowWordCount {

    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1); // 为了尽快触发窗口计算.
        System.out.println("Watermark间隔时间=" + env.getConfig().getAutoWatermarkInterval() + "ms");

        // 设置为1s.
        env.getConfig().setAutoWatermarkInterval(300);

        DataStreamSource<String> sourceDS = env.socketTextStream("master", 9999);

        // 将一行字符串转换成Word对象. 数据格式：
        // word,timestamp
        // hello,1000
        // word,3000
        SingleOutputStreamOperator<Word> wordDS = sourceDS.map((MapFunction<String, Word>) value -> {
            String[] split = value.split(",");
            return new Word(split[0], 1, Long.valueOf(split[1]));
        });

        // 拥有AscendingTimestampsWatermarks的水位线策略.
        WatermarkStrategy<Word> watermarkStrategy = context -> new MyBoundedOutOfOrdernessWatermarks<>(Duration.ofMillis(1000));

        // 自定义实现时间戳分配器.
        watermarkStrategy = watermarkStrategy.withTimestampAssigner((SerializableTimestampAssigner<Word>) (element, recordTimestamp) -> {
            System.out.println("分配时间戳:" + element);
            return element.getTt();
        });

        SingleOutputStreamOperator<Word> watermarkDS = wordDS.assignTimestampsAndWatermarks(watermarkStrategy);

        // 按照word分组.
        KeyedStream<Word, String> keyedStream = watermarkDS.keyBy(new KeySelector<Word, String>() {
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
                System.out.println("窗口window=" + window + "，窗口中数据input=" + input);
            }
        });

        // 按照count列统计和.
        windowDS.sum("count").map(word -> word.getWord() + "=" + word.getCount()).print("-->");

        env.execute();

    }

}
