package flink.application.study.datastream.joining;

import org.apache.flink.api.common.eventtime.*;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.ProcessJoinFunction;
import org.apache.flink.util.Collector;

import java.time.Duration;

public class IntervalJoinDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataStream<People> stream = env.fromData(
                new People("安徽", "zhang", 0L),
                new People("安徽", "li", 2L),
                new People("安徽", "wang", 3L),
                new People("贵州", "qian", 4L),
                new People("贵州", "sun", 5L),
                new People("北京", "duan", 7L)
        ).assignTimestampsAndWatermarks(new WatermarkStrategy<People>() {
            @Override
            public WatermarkGenerator<People> createWatermarkGenerator(WatermarkGeneratorSupplier.Context context) {
                return new AscendingTimestampsWatermarks<>();
            }
        }.withTimestampAssigner(new SerializableTimestampAssigner<People>() {
            @Override
            public long extractTimestamp(People people, long recordTimestamp) {
                return people.timestamp;
            }
        }));

        DataStream<Car> otherStream = env.fromData(
                new Car("安徽", "black", 0L),
                new Car("安徽", "green", 1L),
                new Car("北京", "blue", 6L),
                new Car("北京", "red", 7L)
        ).assignTimestampsAndWatermarks(new WatermarkStrategy<Car>() {
            @Override
            public WatermarkGenerator<Car> createWatermarkGenerator(WatermarkGeneratorSupplier.Context context) {
                return new AscendingTimestampsWatermarks<>();
            }
        }.withTimestampAssigner(new SerializableTimestampAssigner<Car>() {
            @Override
            public long extractTimestamp(Car car, long recordTimestamp) {
                return car.timestamp;
            }
        }));


        KeySelector<People, String> peopleKeySelector = new KeySelector<People, String>() {
            @Override
            public String getKey(People people) throws Exception {
                return people.city;
            }
        };

        KeySelector<Car, String> cartKeySelector = new KeySelector<Car, String>() {
            @Override
            public String getKey(Car car) throws Exception {
                return car.city;
            }
        };

        stream.keyBy(peopleKeySelector).intervalJoin(otherStream.keyBy(cartKeySelector))
                .between(Duration.ofMillis(-2), Duration.ofMillis(1))
//                .upperBoundExclusive()
//                .lowerBoundExclusive()
                .process(new ProcessJoinFunction<People, Car, String>() {
                    @Override
                    public void processElement(People people, Car car, ProcessJoinFunction<People, Car, String>.Context ctx, Collector<String> out) throws Exception {

                        out.collect("左边元素时间戳=" + ctx.getLeftTimestamp() +
                                ",右边元素时间戳=" + ctx.getRightTimestamp() +
                                ",连接结果的时间戳=" + ctx.getTimestamp() +
                                "," + people + "," + car);
                    }
                }).print("|-->");

        env.execute();

    }


    public static class People {
        private String city;
        private String name;
        private Long timestamp;

        public People(String city, String name, Long timestamp) {
            this.city = city;
            this.name = name;
            this.timestamp = timestamp;
        }

        @Override
        public String toString() {
            return "People{" +
                    "city='" + city + '\'' +
                    ", name='" + name + '\'' +
                    ", timestamp=" + timestamp +
                    '}';
        }
    }

    public static class Car {
        private String city;
        private String color;
        private Long timestamp;

        public Car(String city, String color, Long timestamp) {
            this.city = city;
            this.color = color;
            this.timestamp = timestamp;
        }

        @Override
        public String toString() {
            return "Car{" +
                    "city='" + city + '\'' +
                    ", color='" + color + '\'' +
                    ", timestamp=" + timestamp +
                    '}';
        }
    }
}
