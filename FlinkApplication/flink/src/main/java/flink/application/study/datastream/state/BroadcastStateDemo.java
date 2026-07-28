package flink.application.study.datastream.state;

import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.ReadOnlyBroadcastState;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.typeutils.ListTypeInfo;
import org.apache.flink.streaming.api.datastream.*;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.KeyedBroadcastProcessFunction;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class BroadcastStateDemo {
    public static void main(String[] args) throws Exception {
        // 初始化环境.
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 从集合中读取Item，模拟Item数据流
        DataStreamSource<Item> itemDataStreamSource = env.fromData(Arrays.asList(
                new Item(new Color("red"), new Shape("rectangle")),
                new Item(new Color("blue"), new Shape("rectangle")),
                new Item(new Color("red"), new Shape("triangle"))
        ));

        // 将Item的Color作为键.
        KeyedStream<Item, Color> colorPartitionedStream = itemDataStreamSource.keyBy(new KeySelector<Item, Color>() {
            @Override
            public Color getKey(Item value) throws Exception {
                return value.getColor();
            }
        });

        // 从集合中读取Rule，模拟Rule数据流.
        DataStreamSource<Rule> ruleStream = env.fromData(Arrays.asList(
                new Rule("r-t", new Shape("rectangle"), new Shape("triangle")),
                new Rule("t-r", new Shape("triangle"), new Shape("rectangle"))
        ));

        // 初始化映射状态描述器，map的key存储规则名称，value存储规则本身.
        MapStateDescriptor<String, Rule> ruleMapStateDescriptor = new MapStateDescriptor<>(
                "RulesBroadcastState",
                BasicTypeInfo.STRING_TYPE_INFO,
                TypeInformation.of(new TypeHint<Rule>() {
                }));

        // 注意与DataStream<T> broadcast()广播分区的区别，此处广播的是规则映射状态描述器
        BroadcastStream<Rule> ruleBroadcastStream = ruleStream
                .broadcast(ruleMapStateDescriptor);

        BroadcastConnectedStream<Item, Rule> connectedStream = colorPartitionedStream
                .connect(ruleBroadcastStream);

        DataStream<String> output = connectedStream
                .process(new MyKeyedBroadcastProcessFunction());

        output.print("|--->");

        env.execute();
    }

    static class MyKeyedBroadcastProcessFunction extends KeyedBroadcastProcessFunction<Color, Item, Rule, String> {

        // 映射状态描述器，key存储规则名称，value存储Item列表.
        private final MapStateDescriptor<String, List<Item>> mapStateDescriptor =
                new MapStateDescriptor<>(
                        "items",
                        BasicTypeInfo.STRING_TYPE_INFO,
                        new ListTypeInfo<>(Item.class));

        // 与上面定义的规则状态描述器相同.
        private final MapStateDescriptor<String, Rule> ruleMapStateDescriptor =
                new MapStateDescriptor<>(
                        "RulesBroadcastState",
                        BasicTypeInfo.STRING_TYPE_INFO,
                        TypeInformation.of(new TypeHint<Rule>() {
                        }));

        @Override
        public void processElement(Item item, ReadOnlyContext ctx, Collector<String> out) throws Exception {
            // 获取映射状态，key存储规则名称，value存储Item列表.
            final MapState<String, List<Item>> mapState = getRuntimeContext().getMapState(mapStateDescriptor);

            final Shape shape = item.getShape(); // 例如：rectangle

            // 获取广播状态.
            ReadOnlyBroadcastState<String, Rule> broadcastState = ctx.getBroadcastState(ruleMapStateDescriptor);

            // 遍历广播状态.
            for (Map.Entry<String, Rule> entry : broadcastState.immutableEntries()) { // 例如：r-t规则.
                final String ruleName = entry.getKey(); // 例如：r-t.
                final Rule rule = entry.getValue();    // 例如：<rectangle,triangle>

                List<Item> stored = mapState.get(ruleName); // 例如：开始为null.
                if (stored == null) {
                    stored = new ArrayList<>();
                }

                if (shape.equals(rule.second) && !stored.isEmpty()) {
                    for (Item i : stored) {
                        out.collect("MATCH: " + i + " - " + item);
                    }
                    stored.clear();
                }

                if (shape.equals(rule.first)) { // 例如：满足条件
                    stored.add(item); // 把<red,rectangle>的item添加进去
                }

                if (stored.isEmpty()) {
                    mapState.remove(ruleName);
                } else {
                    mapState.put(ruleName, stored); // <r-t,<red,rectangle>> 添加进去.
                }
            }
        }

        @Override
        public void processBroadcastElement(Rule rule, Context ctx, Collector<String> out) throws Exception {
            ctx.getBroadcastState(ruleMapStateDescriptor).put(rule.name, rule);
        }
    }

}
