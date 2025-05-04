package de.ude.ies.elastic_ai.controller;

import de.ude.ies.elastic_ai.entities.User;
import de.ude.ies.elastic_ai.repositories.UserRepository;
import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.UUID;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping({ "/user" })
@Slf4j
public class UserController {

    private final UserRepository users;

    public UserController(final UserRepository users) {
        this.users = users;
    }

    @GetMapping({ "", "/", "/index", "/index.html" })
    public String index(Model model) {
        ArrayList<User> userList = new ArrayList<>();
        users.findAll().forEach(userList::add);
        model.addAttribute("users", userList);
        model.addAttribute("newUser", new User());
        return "user/index";
    }

    @PostMapping("/add")
    public String add(@ModelAttribute User newUser) {
        users.save(newUser);
        log.info("User added: {}", newUser);
        return "redirect:/user";
    }

    @GetMapping("/{id}")
    public String userInfo(@PathVariable("id") UUID id, Model model) {
        User user = users.findById(id).orElseThrow(NoSuchElementException::new);
        model.addAttribute("user", user);
        return "user/info";
    }

    @PostMapping("/delete/{id}")
    public String delete(@PathVariable("id") UUID id) {
        users.deleteById(id);
        log.info("User deleted: {}", id);
        return "redirect:/user";
    }

    @PostMapping("/update/{id}")
    public String update(@ModelAttribute User user, @PathVariable("id") UUID id)
        throws NoSuchElementException {
        users
            .findById(id)
            .ifPresent(oldUser -> {
                oldUser.setName(user.getName());
                oldUser.setArmLength(user.getArmLength());
                oldUser.setShoulderHeight(user.getShoulderHeight());
                oldUser.setDominantHand(user.getDominantHand());
                users.save(oldUser);
            });
        log.info("User updated: {}", user);
        return "redirect:/user";
    }
}
