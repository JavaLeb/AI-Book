package flink.application.study.datastream.window;

import org.apache.flink.api.common.eventtime.AscendingTimestampsWatermarks;
import org.apache.flink.api.common.eventtime.SerializableTimestampAssigner;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.WindowedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.SlidingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

import java.time.Duration;

public class SlidingWindow {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        // 从集合读取数据.
        DataStream<User> source = env.fromData(
                new User("user1", 1000L, 1),
                new User("user1", 2000L, 2),
                new User("user1", 3000L, 3),
                new User("user2", 4000L, 3),
                new User("user3", 5000L, 4)
        );

        // watermark.
        WatermarkStrategy<User> watermarkStrategy = context -> new AscendingTimestampsWatermarks<>();
        SerializableTimestampAssigner<User> assigner = (element, recordTimestamp) -> element.getTimestamp();
        DataStream<User> watermarksDS = source.assignTimestampsAndWatermarks(watermarkStrategy.withTimestampAssigner(assigner));

        //Keyed Stream，定义滚动窗口.
        WindowedStream<User, String, TimeWindow> windowedStream =
                watermarksDS.keyBy(User::getName).window(SlidingEventTimeWindows.of(Duration.ofMillis(3000),Duration.ofMillis(1000)));

        windowedStream.process(new ProcessWindowFunction<User, String, String, TimeWindow>() {
            @Override
            public void process(String key, ProcessWindowFunction<User, String, String, TimeWindow>.Context context, Iterable<User> elements, Collector<String> out) throws Exception {
                int totalAge = 0;
                for (User user : elements) {
                    totalAge += user.getValue();
                }
                TimeWindow timeWindow = context.window();
                String windowInfo = "Window[" + timeWindow.getStart() + "," + timeWindow.getEnd() + ") | "
                        + "maxTimestamp=" + timeWindow.maxTimestamp() +
                        ", key=" + key + "," + " totalAge=" + totalAge + " |  elements=" + elements;
                out.collect(windowInfo);
            }
        }).print("|-->");

        env.execute();

    }
}
