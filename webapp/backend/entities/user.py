"""Instance of a User Entity."""

from dataclasses import dataclass, field
from enum import Enum
from random import choices
from string import ascii_letters, digits
from uuid import UUID, uuid1


class DominantHand(str, Enum):
    """Dominant Hand of User."""

    LEFT = "L"
    RIGHT = "R"


@dataclass
class User:
    """Entity representing a User."""

    @staticmethod
    def __generate_password() -> str:
        return "".join(choices(ascii_letters + digits, k=12))

    name: str | None = None
    arm_length: int | None = None
    shoulder_height: int | None = None
    dominant_hand: DominantHand | None = None
    password: str = field(default_factory=__generate_password)
    uid: UUID = field(default_factory=uuid1)

    def valid_user(self) -> bool:
        """Check if all fields are set to form a valid User."""
        return all(
            [
                self.name,
                self.arm_length,
                self.shoulder_height,
                self.dominant_hand,
                self.password,
                self.uid,
            ]
        )

    def __str__(self) -> str:
        return str(self.name) + "(" + str(self.uid) + ")"

    def __eq__(self, other) -> bool:
        return self.uid == other.uid

    def __hash__(self) -> int:
        return hash(str(self))


if __name__ == "__main__":
    usr = User("David", 45, 175, DominantHand.RIGHT)
    print(usr.valid_user())
