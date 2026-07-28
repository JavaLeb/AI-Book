package flink.application.study.datastream.watermark;

import org.apache.flink.api.common.eventtime.Watermark;
import org.apache.flink.api.common.eventtime.WatermarkGenerator;
import org.apache.flink.api.common.eventtime.WatermarkOutput;

public class PunctuatedWatermarkGenerator<T> implements WatermarkGenerator<T> {
    private Long maxTimestamp;

    @Override
    public void onEvent(T event, long eventTimestamp, WatermarkOutput output) {
        maxTimestamp = Math.max(maxTimestamp, eventTimestamp);
        output.emitWatermark(new Watermark(maxTimestamp));
    }

    @Override
    public void onPeriodicEmit(WatermarkOutput output) {

    }
}
