package flink.application.study.datastream.window;

import org.apache.flink.api.common.eventtime.AscendingTimestampsWatermarks;
import org.apache.flink.api.common.eventtime.SerializableTimestampAssigner;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.WindowedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;

import java.time.Duration;

public class ReduceFunctionDemo {
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
                watermarksDS.keyBy(User::getName).window(TumblingEventTimeWindows.of(Duration.ofMillis(3000)));


        windowedStream.reduce(
                new ReduceFunction<User>() {
                    @Override
                    public User reduce(User user1, User user2) throws Exception {
                        return new User(user1.getName(),
                                Math.max(user1.getTimestamp(), user2.getTimestamp()),
                                user1.getValue() + user2.getValue());
                    }
                }
        ).print("|-->");

        env.execute();
    }
}
