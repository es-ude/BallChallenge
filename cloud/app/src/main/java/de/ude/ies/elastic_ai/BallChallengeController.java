package de.ude.ies.elastic_ai;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping({ "" })
public class BallChallengeController {

    @Autowired
    BallChallengeEndpoint endpoint;

    @GetMapping({ "/", "/index", "/index.html" })
    public String clientLandingPage(Model model) {
        model.addAttribute("abc", endpoint);
        return "index";
    }

    @PostMapping("/start")
    public ResponseEntity<Object> startMeasurement() {
        endpoint.publishStartMeasurement();
        return ResponseEntity.status(HttpStatus.OK).build();
    }

    @PostMapping("/setID")
    public ResponseEntity<Object> setEnV5ID(String id) {
        endpoint.setEnV5ID(id);
        return ResponseEntity.status(HttpStatus.OK).build();
    }

    @GetMapping("/requestCountDownUpdate")
    @ResponseBody
    public DataValue requestCountDownUpdate() {
        return new DataValue(endpoint.getLastTime());
    }

    @GetMapping("/requestGValueUpdate")
    @ResponseBody
    public DataValue requestData() {
        return new DataValue(endpoint.getLastGValue());
    }

    public record DataValue(String VALUE) {}
}
