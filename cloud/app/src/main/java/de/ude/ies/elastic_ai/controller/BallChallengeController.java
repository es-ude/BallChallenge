package de.ude.ies.elastic_ai.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping({ "" })
public class BallChallengeController {

    @GetMapping({ "", "/", "/index", "/index.html" })
    public String index() {
        return "index";
    }
}
