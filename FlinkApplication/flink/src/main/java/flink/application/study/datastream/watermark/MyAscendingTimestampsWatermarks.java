package flink.application.study.datastream.watermark;

import org.apache.flink.api.common.eventtime.Watermark;
import org.apache.flink.api.common.eventtime.WatermarkGenerator;
import org.apache.flink.api.common.eventtime.WatermarkOutput;

public class MyAscendingTimestampsWatermarks<T> implements WatermarkGenerator<T> {
    private long maxTimestamp = Long.MIN_VALUE + 1;

    @Override
    public void onEvent(T event, long eventTimestamp, WatermarkOutput output) {
        System.out.println("Watermark生成器处理事件:" + event);
        maxTimestamp = Math.max(maxTimestamp, eventTimestamp);
    }

    @Override
    public void onPeriodicEmit(WatermarkOutput output) {
        Watermark watermark = new Watermark(maxTimestamp - 1);
        // 默认每隔200ms打印.
//        System.out.println("发送Watermark:" + watermark);
        output.emitWatermark(watermark);
    }
}
