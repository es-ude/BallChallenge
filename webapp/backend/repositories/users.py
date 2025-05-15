"""List of User Entities."""

from itertools import islice
from typing import Callable
from uuid import UUID

import orjson as json

from webapp.backend.entities.user import DominantHand, User


class DatasetError(Exception):
    """Error updating the data repository."""

    pass


class Users:
    """User repository."""

    _users: dict[UUID, User]
    _path: str
    _autosave_enabled: bool

    @staticmethod
    def __autosave(func: Callable) -> Callable:
        def decorator(
            self,  # noqa: ANN001
            *argv: str,
            **kwargs: int,
        ) -> None:
            func(self, *argv, **kwargs)
            if self._autosave_enabled:
                self.save()

        return decorator

    @staticmethod
    def __validate_user(func: Callable) -> Callable:
        def decorator(
            self,  # noqa: ANN001
            user: User,
        ) -> None:
            if not isinstance(user, User) or not user.valid_user():
                raise DatasetError("Invalid User!")
            func(self, user)

        return decorator

    def __init__(
        self, path: str = "settings/users.json", autosave_enabled: bool = True
    ) -> None:
        """Initialise instance of class."""
        self._users = {}
        self._path = path
        self._autosave_enabled = autosave_enabled
        self.load()

    def save(self) -> None:
        """Persist user database."""
        with open(self._path, "wb") as user_file:
            user_file.write(
                json.dumps(
                    list(self._users.values()),
                    option=json.OPT_INDENT_2 | json.OPT_APPEND_NEWLINE,
                )
            )

    def load(self) -> None:
        """Read user database."""
        try:
            with open(self._path, "rb") as user_file:
                users = json.loads(user_file.read())
            self._users.clear()
            for user in users:
                self.insert(
                    User(
                        name=user["name"],
                        arm_length=user["arm_length"],
                        shoulder_height=user["shoulder_height"],
                        dominant_hand=(
                            DominantHand.RIGHT
                            if user["dominant_hand"] == "R"
                            else DominantHand.LEFT
                        ),
                        password=user["password"],
                        uid=UUID(user["uid"]),
                    )
                )
        except FileNotFoundError:
            self.save()

    @__autosave
    @__validate_user
    def insert(self, user: User) -> None:
        """Add user to list.

        Args:
            user (User): user to add

        Raises:
            DatasetError if User exists already
                         if User instance is invalid

        Returns:
            None
        """
        if user.uid in self._users:
            raise DatasetError("User already existing!")
        if user.name in [user.name for user in self._users.values()]:
            raise DatasetError("User name already taken!")
        self._users[user.uid] = user

    @__autosave
    @__validate_user
    def update(self, user: User) -> None:
        """Update existing user."""
        self._users[user.uid] = user

    @__autosave
    @__validate_user
    def delete(self, user: User) -> None:
        """Remove user from database."""
        self._users.pop(user.uid)

    def find_all(self, limit: int = -1) -> dict[UUID, str]:
        """Get immutable list with all Users.

        Args:
            limit (int): max number of elements to return
                         limit<0 -> all elements

        Returns:
            set of User
        """
        if limit < 0:
            return dict(
                [
                    (user.uid, "Anonym" if user.name is None else user.name)
                    for user in self._users.values()
                ]
            )
        else:
            return dict(
                [
                    (user.uid, "Anonym" if user.name is None else user.name)
                    for user in islice(self._users.values(), limit)
                ]
            )

    def find_by_uuid(self, uid: UUID | str) -> User | None:
        """Get User by UUID.

        Args:
            uid (UUID | str): UUID to find user for

        Raises:
            ValueError: if uid is invalid

        Returns:
            user if present, None else
        """
        user_id: UUID = UUID(uid) if isinstance(uid, str) else uid
        return self._users.get(user_id, None)

    def __str__(self) -> str:
        """Generate String representation of class."""
        users: str = "[\n"
        for uid, user in self._users.items():
            users += "\t" + str(uid) + ": " + str(user) + "\n"
        users += "]"
        return users


if __name__ == "__main__":
    users = Users()
    print(str(users))
