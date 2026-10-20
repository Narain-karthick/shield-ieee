package ai.shieldstream;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public final class BaselinePipeline {
    private static final ObjectMapper JSON = new ObjectMapper();

    private BaselinePipeline() {}

    public static void main(String[] args) throws Exception {
        String bootstrap = env("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092");
        String input = env("RAW_TOPIC", "taxi-raw");
        String clean = env("CLEAN_TOPIC", "taxi-clean");
        String dlq = env("DLQ_TOPIC", "taxi-dlq");

        StreamExecutionEnvironment environment = StreamExecutionEnvironment.getExecutionEnvironment();
        environment.enableCheckpointing(10_000, CheckpointingMode.EXACTLY_ONCE);

        KafkaSource<String> source = KafkaSource.<String>builder()
                .setBootstrapServers(bootstrap)
                .setTopics(input)
                .setGroupId("shield-baseline")
                .setStartingOffsets(OffsetsInitializer.earliest())
                .setValueOnlyDeserializer(new SimpleStringSchema())
                .build();

        DataStream<String> events = environment.fromSource(source, WatermarkStrategy.noWatermarks(), "taxi-source");
        DataStream<String> valid = events.filter(BaselinePipeline::isValid).name("schema-and-quality-validator");
        DataStream<String> invalid = events.filter(value -> !isValid(value)).name("invalid-event-router");

        valid.sinkTo(sink(bootstrap, clean)).name("clean-topic");
        invalid.sinkTo(sink(bootstrap, dlq)).name("dead-letter-topic");
        environment.execute("SHIELD-Stream Baseline Pipeline");
    }

    static boolean isValid(String value) {
        try {
            JsonNode event = JSON.readTree(value);
            return event.hasNonNull("event_id")
                    && event.path("schema_version").asInt(-1) == 1
                    && event.hasNonNull("trip_distance")
                    && event.path("trip_distance").isNumber()
                    && event.path("trip_distance").asDouble(-1) >= 0
                    && event.hasNonNull("fare_amount")
                    && event.path("fare_amount").isNumber()
                    && event.path("passenger_count").asDouble(0) >= 0;
        } catch (Exception ignored) {
            return false;
        }
    }

    static KafkaSink<String> sink(String bootstrap, String topic) {
        return KafkaSink.<String>builder()
                .setBootstrapServers(bootstrap)
                .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                        .setTopic(topic)
                        .setValueSerializationSchema(new SimpleStringSchema())
                        .build())
                .build();
    }

    static String env(String name, String fallback) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? fallback : value;
    }
}

