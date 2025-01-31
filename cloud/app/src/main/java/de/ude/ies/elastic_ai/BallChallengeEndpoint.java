package de.ude.ies.elastic_ai;

import static de.ude.ies.elastic_ai.protocol.Status.State.ONLINE;

import de.ude.ies.elastic_ai.communicationEndpoints.LocalCommunicationEndpoint;
import de.ude.ies.elastic_ai.communicationEndpoints.RemoteCommunicationEndpoint;
import de.ude.ies.elastic_ai.protocol.Status;
import de.ude.ies.elastic_ai.protocol.requests.DataRequester;
import java.io.*;
import java.net.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.Timestamp;
import java.util.*;
import lombok.Getter;

public class BallChallengeEndpoint extends LocalCommunicationEndpoint {

    private final String CameraIp;

    private final Integer CameraPort;

    private RemoteCommunicationEndpoint enV5;
    private final String DATA_PATH = "SensorValues";

    @Getter
    private final Set<String> enV5IDs = new HashSet<>();

    @Getter
    private String lastTime = "";

    @Getter
    private String lastGValue = "";

    private DataRequester dataRequesterAccelerometer;
    private DataRequester dataRequesterTime;

    public BallChallengeEndpoint(
        String PublicIp,
        Integer PublicPort,
        String CameraIp,
        Integer CameraPort
    ) {
        super("ballChallengeApplication", "APPLICATION");
        this.status.ADD_OPTIONAL("WEBSITE", PublicIp + ":" + PublicPort);

        this.CameraIp = CameraIp;
        this.CameraPort = CameraPort;

        File sensorValueDir = new File(DATA_PATH);
        createFolder(sensorValueDir);
    }

    @Override
    protected void executeOnBind() {
        RemoteCommunicationEndpoint statusReceiver = new RemoteCommunicationEndpoint("+");
        statusReceiver.bindToCommunicationEndpoint(broker);
        statusReceiver.subscribeForStatus(posting -> {
            if (Objects.equals(Status.extractFromStatus(posting.data(), "TYPE"), "enV5")) {
                if (
                    Objects.equals(
                        Status.extractFromStatus(posting.data(), "STATE"),
                        ONLINE.toString()
                    )
                ) {
                    enV5IDs.add(Status.extractFromStatus(posting.data(), "ID"));
                } else {
                    enV5IDs.remove(Status.extractFromStatus(posting.data(), "ID"));
                }
            }
        });
    }

    public void setEnV5ID(String id) {
        if (enV5 != null && Objects.equals(enV5.getIdentifier(), id)) {
            return;
        }

        resetStub();

        if (Objects.equals(id, "")) {
            return;
        }

        createStub(id);
    }

    private void createStub(String id) {
        enV5 = new RemoteCommunicationEndpoint(id);
        enV5.bindToCommunicationEndpoint(broker);
        dataRequesterAccelerometer = new DataRequester(
            enV5,
            "acceleration",
            getDomainAndIdentifier()
        );
        dataRequesterAccelerometer.setDataReceiveFunction(handleThrowData());
        dataRequesterAccelerometer.listenToData(true);

        dataRequesterTime = new DataRequester(enV5, "timer", getDomainAndIdentifier());
        dataRequesterTime.setDataReceiveFunction(data -> lastTime = data);
        dataRequesterTime.listenToData(true);
    }

    private void resetStub() {
        enV5 = null;
        lastGValue = "";
        if (dataRequesterAccelerometer != null) {
            dataRequesterAccelerometer.listenToData(false);
        }
        lastTime = "";
        if (dataRequesterTime != null) {
            dataRequesterTime.listenToData(false);
        }
    }

    public void publishStartMeasurement() {
        if (enV5 == null) {
            return;
        }
        enV5.publishCommand("MEASUREMENT", this.identifier);
    }

    private void savePicture(String filePath) {
        // Clear camera buffer
        for (int i = 0; i < 10; i++) {
            try (
                InputStream ignored = new URI("http://" + CameraIp + ":" + CameraPort + "/jpeg")
                    .toURL()
                    .openStream()
            ) {
                Thread.sleep(10);
            } catch (IOException | InterruptedException | URISyntaxException e) {
                throw new RuntimeException(e);
            }
        }

        // Take picture
        try (
            InputStream in = new URI("http://" + CameraIp + ":" + CameraPort + "/jpeg")
                .toURL()
                .openStream()
        ) {
            Files.copy(in, Paths.get(filePath + "/image.jpg"));
        } catch (MalformedURLException | URISyntaxException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            System.out.println("NO PICTURE TAKEN!!!");
        }
    }

    private DataExecutor handleThrowData() {
        return data -> {
            System.out.println("Handling data");
            lastGValue = data.split(";")[0];
            String timeStamp = new Timestamp(System.currentTimeMillis())
                .toString()
                .split("\\.")[0].replace(":", "-")
                .replace(" ", "_");
            String folderName = DATA_PATH + "/" + timeStamp;
            try {
                System.out.println("Saving throw to: " + folderName);

                File csvDir = new File(folderName);
                createFolder(csvDir);

                savePicture(folderName);

                FileWriter csvWriter = new FileWriter(folderName + "/measurement.csv", false);

                csvWriter
                    .append("x")
                    .append(",")
                    .append("y")
                    .append(",")
                    .append("z")
                    .append("\n")
                    .append(data.replace(";", "\n"));
                csvWriter.flush();
                csvWriter.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        };
    }

    private void createFolder(File folder) {
        if (folder.isFile()) {
            throw new RuntimeException("Folder already exists and is a file!");
        }
        if (folder.isDirectory()) {
            return;
        }
        if (!folder.mkdirs()) {
            throw new RuntimeException("Can't create folder to store sensor values!");
        }
    }
}
