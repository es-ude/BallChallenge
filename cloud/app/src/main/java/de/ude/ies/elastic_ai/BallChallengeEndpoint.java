package de.ude.ies.elastic_ai;

import static de.ude.ies.elastic_ai.protocol.Status.State.ONLINE;

import de.ude.ies.elastic_ai.communicationEndpoints.LocalCommunicationEndpoint;
import de.ude.ies.elastic_ai.communicationEndpoints.RemoteCommunicationEndpoint;
import de.ude.ies.elastic_ai.entities.User;
import de.ude.ies.elastic_ai.protocol.Posting;
import de.ude.ies.elastic_ai.protocol.Status;
import de.ude.ies.elastic_ai.protocol.requests.DataRequester;
import java.io.*;
import java.net.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class BallChallengeEndpoint extends LocalCommunicationEndpoint {

    private final String CameraIp;

    private final Integer CameraPort;

    private RemoteCommunicationEndpoint enV5;

    @Setter
    private User user;

    private final String DATA_PATH = "SensorValues";

    @Getter
    private final Set<String> enV5IDs = new HashSet<>();

    DateTimeFormatter timestampFormat = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");

    @Getter
    private String countdown = "";

    @Getter
    private String[] lastMeasurement = null;

    private DataRequester dataRequesterAccelerometer;
    private DataRequester dataRequesterCountdown;

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
        setupStatusReceiver();
    }

    // region DEVICE LIST

    private void setupStatusReceiver() {
        RemoteCommunicationEndpoint statusReceiver = new RemoteCommunicationEndpoint("+");
        statusReceiver.bindToCommunicationEndpoint(broker);
        statusReceiver.subscribeForStatus(this::statusHandler);
    }

    private void statusHandler(Posting posting) {
        if (Status.extractFromStatus(posting.data(), "TYPE").equals("enV5")) {
            String state = Status.extractFromStatus(posting.data(), "STATE");
            String id = Status.extractFromStatus(posting.data(), "ID");

            if (state.equals(ONLINE.toString())) {
                enV5IDs.add(id);
            } else {
                enV5IDs.remove(id);
            }
        }
    }

    // endregion DEVICE LIST

    // region UPDATE NODE/USER

    public void setEnV5ID(String id) {
        if (enV5 == null || !enV5.getIdentifier().equals(id)) {
            resetStub();
            resetValueStore();
            if (!id.isEmpty()) {
                createStub(id);
            }
        }
    }

    private void resetValueStore() {
        lastMeasurement = null;
        countdown = null;
    }

    private void resetStub() {
        enV5 = null;
        if (dataRequesterAccelerometer != null) {
            dataRequesterAccelerometer.listenToData(false);
        }
        if (dataRequesterCountdown != null) {
            dataRequesterCountdown.listenToData(false);
        }
    }

    private void createStub(String id) {
        enV5 = new RemoteCommunicationEndpoint(id);
        enV5.bindToCommunicationEndpoint(broker);
        setupDataRequesterAccelerometer();
        setupDataRequesterTimer();
    }

    private void setupDataRequesterAccelerometer() {
        dataRequesterAccelerometer = new DataRequester(
            enV5,
            "acceleration",
            getDomainAndIdentifier()
        );
        dataRequesterAccelerometer.setDataReceiveFunction(new ThrowHandler());
        dataRequesterAccelerometer.listenToData(true);
    }

    private void setupDataRequesterTimer() {
        dataRequesterCountdown = new DataRequester(enV5, "timer", getDomainAndIdentifier());
        dataRequesterCountdown.setDataReceiveFunction(data -> countdown = data);
        dataRequesterCountdown.listenToData(true);
    }

    // endregion UPDATE NODE/USER

    // region TAKE MEASUREMENT

    public void startMeasurement() {
        if (enV5 == null) {
            throw new IllegalStateException("No enV5 available");
        }
        countdown = null;
        lastMeasurement = null;
        enV5.publishCommand("MEASUREMENT", this.identifier);
    }

    private class ThrowHandler implements DataExecutor {

        @Override
        public void execute(String data) {
            log.info("Handling datasource");
            String folderName = DATA_PATH + "/" + timestampFormat.format(LocalDateTime.now());
            createFolder(new File(folderName));
            saveParticipant(folderName, user);
            saveData(folderName, data);
            savePicture(folderName);
        }
    }

    private void createFolder(File folder) {
        if (folder.isFile()) {
            throw new RuntimeException("Folder already exists and is a file!");
        } else if (folder.isDirectory()) {} else if (!folder.mkdirs()) {
            throw new RuntimeException("Can't create folder to store sensor values!");
        }
    }

    private void savePicture(String filePath) {
        try {
            clearCameraBuffer();
            takePicture(filePath);
        } catch (RuntimeException | IOException e) {
            log.error(e.getMessage());
            log.warn("No picture taken!");
        }
    }

    private void clearCameraBuffer() throws RuntimeException {
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
    }

    private void takePicture(String filePath) throws RuntimeException, IOException {
        try (
            InputStream in = new URI("http://" + CameraIp + ":" + CameraPort + "/jpeg")
                .toURL()
                .openStream()
        ) {
            Files.copy(in, Paths.get(filePath + "/image.jpg"));
        } catch (MalformedURLException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    private void saveParticipant(String folderName, User participant) {
        try (FileWriter csvWriter = new FileWriter(folderName + "/user.csv", false)) {
            log.info("Saving user to '{}/user.csv'", folderName);
            csvWriter
                .append("name")
                .append(", ")
                .append("shoulder height")
                .append(", ")
                .append("arm length")
                .append(", ")
                .append("dominant Hand")
                .append("\n");
            csvWriter.append(participant.getName()).append(", ");
            csvWriter.append(String.valueOf(participant.getShoulderHeight())).append(", ");
            csvWriter.append(String.valueOf(participant.getArmLength())).append(", ");
            csvWriter.append(participant.getDominantHand().name()).append("\n");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void saveData(String folderName, String data) {
        lastMeasurement = data.split(";");
        try (FileWriter csvWriter = new FileWriter(folderName + "/measurement.csv", false)) {
            log.info("Saving data to '{}/measurement.csv'", folderName);
            csvWriter.append("x").append(", ").append("y").append(", ").append("z").append("\n");
            for (String sample : lastMeasurement) {
                csvWriter.append(sample.strip()).append("\n");
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    // endregion TAKE MEASUREMENT
}
