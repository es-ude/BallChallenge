package de.ude.ies.elastic_ai.controller;

import static java.lang.Thread.sleep;

import de.ude.ies.elastic_ai.BallChallengeEndpoint;
import de.ude.ies.elastic_ai.entities.User;
import de.ude.ies.elastic_ai.repositories.UserRepository;
import java.util.ArrayList;
import java.util.UUID;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping({ "/dc" })
@Slf4j
public class DataCollectController {

    public record DataValue(String VALUE) {}

    @NoArgsConstructor
    @Getter
    @Setter
    public static class StartRequest {

        private String node;
        private UUID user;
    }

    private final BallChallengeEndpoint endpoint;

    private final UserRepository users;

    public DataCollectController(final BallChallengeEndpoint endpoint, final UserRepository users) {
        this.endpoint = endpoint;
        this.users = users;
    }

    @GetMapping({ "", "/", "/index", "/index.html" })
    public String index(Model model) {
        ArrayList<User> userList = new ArrayList<>();
        users.findAll().forEach(userList::add);
        model.addAttribute("participants", userList);
        model.addAttribute("nodes", endpoint.getEnV5IDs());
        model.addAttribute("startRequest", new StartRequest());
        return "dc/index";
    }

    @PostMapping("/start")
    public String startMeasurement(@ModelAttribute StartRequest startRequest) {
        if (startRequest.node == null || startRequest.user == null) {
            log.warn("Can't start measurement because node or user is null!");
            return "redirect:/dc/index";
        }

        if (!endpoint.getEnV5IDs().contains(startRequest.node)) {
            log.warn(
                "Can't start measurement because node '{}' is not available!",
                startRequest.node
            );
            return "redirect:/dc/index";
        }
        endpoint.setEnV5ID(startRequest.node);

        if (users.findById(startRequest.user).isEmpty()) {
            log.warn(
                "Can't start measurement because user '{}' does not exist!",
                startRequest.user
            );
            return "redirect:/dc/index";
        }
        endpoint.setUser(users.findById(startRequest.user).get());

        log.info(
            "Start measurement for node '{}' with user '{}'",
            startRequest.node,
            startRequest.user
        );
        endpoint.startMeasurement();
        return "dc/countdown";
    }

    @GetMapping("/countdown")
    @ResponseBody
    public DataValue requestCountDownUpdate() {
        String countdown = endpoint.getCountdown();
        log.info("Requested countdown {}", countdown);
        return new DataValue(countdown);
    }

    @GetMapping("/gvalue")
    @ResponseBody
    public DataValue requestData() {
        String[] data = endpoint.getLastMeasurement();
        if (data == null) {
            log.info("Requested value not available!");
            return new DataValue(null);
        }
        log.info("Requested value {}", data[0]);
        return new DataValue(data[0]);
    }
}
