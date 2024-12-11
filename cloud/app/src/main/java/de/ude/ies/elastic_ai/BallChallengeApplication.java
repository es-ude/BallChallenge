package de.ude.ies.elastic_ai;

import de.ude.ies.elastic_ai.protocol.HivemqBroker;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Collections;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentGroup;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class BallChallengeApplication {

    public static String HOST_IP;
    public static Integer PORT;
    private static final String MQTT_DOMAIN = "eip://uni-due.de/es";
    public static String BROKER_IP = null;
    public static Integer BROKER_PORT = null;
    public static String CAMERA_IP = null;
    public static Integer CAMERA_PORT = null;
    public static BallChallengeEndpoint ballChallengeEndpoint = null;

    public static void main(String[] args) {
        HOST_IP = System.getenv("HOST_IP");
        if (HOST_IP == null) {
            try (final DatagramSocket socket = new DatagramSocket()) {
                socket.connect(InetAddress.getByName("8.8.8.8"), 10002);
                HOST_IP = socket.getLocalAddress().getHostAddress();
            } catch (UnknownHostException | SocketException e) {
                throw new RuntimeException(e);
            }
        }
        HOST_IP = HOST_IP.strip();

        try {
            Namespace arguments = BallChallengeApplication.parseArguments(args);
            BROKER_IP = arguments.getString("broker_address");
            BROKER_PORT = arguments.getInt("broker_port");
            PORT = arguments.getInt("port");
            CAMERA_IP = arguments.getString("camera_address");
            CAMERA_PORT = arguments.getInt("camera_port");
        } catch (ArgumentParserException exception) {
            System.out.println(exception.getMessage());
            System.exit(10);
        }

        ballChallengeEndpoint = new BallChallengeEndpoint(CAMERA_IP, CAMERA_PORT);
        ballChallengeEndpoint.bindToCommunicationEndpoint(
            new HivemqBroker(MQTT_DOMAIN, BROKER_IP, BROKER_PORT)
        );

        SpringApplication app = new SpringApplication(BallChallengeApplication.class);
        app.setDefaultProperties(Collections.singletonMap("server.port", PORT));
        app.run(args);
    }

    static Namespace parseArguments(String[] args) throws ArgumentParserException {
        ArgumentParser parser = ArgumentParsers.newFor("ball-challenge")
            .build()
            .defaultHelp(true)
            .description("Service for controlling the wrist band to collect data-samples");
        parsePort(parser);
        parseBrokerArguments(parser);
        parseCameraArguments(parser);
        return parser.parseKnownArgs(args, null);
    }

    private static void parsePort(ArgumentParser parser) {
        ArgumentGroup brokerSpecification = parser.addArgumentGroup("Web Frontend Specification");

        brokerSpecification
            .addArgument("--port")
            .help("Port to publish website")
            .type(Integer.class)
            .setDefault(80);
    }

    private static void parseBrokerArguments(ArgumentParser parser) {
        ArgumentGroup brokerSpecification = parser.addArgumentGroup("MQTT Broker Specification");

        brokerSpecification
            .addArgument("--broker-address")
            .help("IP Address of the MQTT Broker")
            .setDefault("localhost");
        brokerSpecification
            .addArgument("--broker-port")
            .type(Integer.class)
            .help("Port of the MQTT Broker")
            .setDefault(1883);
    }

    private static void parseCameraArguments(ArgumentParser parser) {
        ArgumentGroup cameraSpecification = parser.addArgumentGroup("Camera Specification");
        cameraSpecification
            .addArgument("--camera-address")
            .help("IP address to connect to webcam")
            .setDefault("localhost");
        cameraSpecification
            .addArgument("--camera-port")
            .help("Port to connect to Camera")
            .type(Integer.class)
            .setDefault(8081);
    }
}
