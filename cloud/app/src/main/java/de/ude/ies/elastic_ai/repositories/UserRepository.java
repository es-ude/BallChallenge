package de.ude.ies.elastic_ai.repositories;

import de.ude.ies.elastic_ai.entities.User;
import java.util.UUID;
import org.springframework.data.repository.CrudRepository;

public interface UserRepository extends CrudRepository<User, UUID> {}
