"""Unit-Tests for User Repository."""

from uuid import UUID

import pytest

from webapp.backend.entities.user import DominantHand, User
from webapp.backend.repositories.users import Users

STORAGE_PATH: str = "storage/users.json"


@pytest.fixture
def users() -> Users:
    return Users(path=STORAGE_PATH)


def test_store_single_users(users: Users, mocker) -> None:
    u1: User = User(
        "USR1",
        45,
        145,
        DominantHand.RIGHT,
        password="7nn6wNWBdh2X",
        uid=UUID("42e68894-30fb-11f0-9449-2dce6e7d0d6d"),
    )
    users.add(u1)

    file_mock = mocker.patch(
        "webapp.backend.repositories.users.open", mocker.mock_open()
    )
    users.store()
    file_mock.assert_called_once_with(STORAGE_PATH, "wb")
    file_mock().write.assert_called_once_with(
        b"[\n"
        + b"  {\n"
        + b'    "name": "USR1",\n'
        + b'    "arm_length": 45,\n'
        + b'    "shoulder_height": 145,\n'
        + b'    "dominant_hand": "R",\n'
        + b'    "password": "7nn6wNWBdh2X",\n'
        + b'    "uid": "42e68894-30fb-11f0-9449-2dce6e7d0d6d"\n'
        + b"  }\n"
        + b"]\n"
    )


def test_store_multiple_users(users: Users, mocker) -> None:
    u1: User = User(
        "USR1",
        45,
        145,
        DominantHand.RIGHT,
        password="7nn6wNWBdh2X",
        uid=UUID("42e68894-30fb-11f0-9449-2dce6e7d0d6d"),
    )
    users.add(u1)
    u2: User = User(
        "USR2",
        55,
        155,
        DominantHand.LEFT,
        password="7nn6wNWBdh2x",
        uid=UUID("42e68894-30fb-11f0-9449-2dce6e7d0d6f"),
    )
    users.add(u2)

    file_mock = mocker.patch(
        "webapp.backend.repositories.users.open", mocker.mock_open()
    )
    users.store()
    file_mock.assert_called_once_with(STORAGE_PATH, "wb")
    file_mock().write.assert_called_once_with(
        b"[\n"
        + b"  {\n"
        + b'    "name": "USR1",\n'
        + b'    "arm_length": 45,\n'
        + b'    "shoulder_height": 145,\n'
        + b'    "dominant_hand": "R",\n'
        + b'    "password": "7nn6wNWBdh2X",\n'
        + b'    "uid": "42e68894-30fb-11f0-9449-2dce6e7d0d6d"\n'
        + b"  },\n"
        + b"  {\n"
        + b'    "name": "USR2",\n'
        + b'    "arm_length": 55,\n'
        + b'    "shoulder_height": 155,\n'
        + b'    "dominant_hand": "L",\n'
        + b'    "password": "7nn6wNWBdh2x",\n'
        + b'    "uid": "42e68894-30fb-11f0-9449-2dce6e7d0d6f"\n'
        + b"  }\n"
        + b"]\n"
    )


def test_load_single_user(users: Users, mocker) -> None:
    u1_json = (
        b"[\n"
        + b"  {\n"
        + b'    "name": "USR1",\n'
        + b'    "arm_length": 45,\n'
        + b'    "shoulder_height": 145,\n'
        + b'    "dominant_hand": "R",\n'
        + b'    "password": "7nn6wNWBdh2X",\n'
        + b'    "uid": "42e68894-30fb-11f0-9449-2dce6e7d0d6d"\n'
        + b"  }\n"
        + b"]\n"
    )
    u1: User = User(
        "USR1",
        45,
        145,
        DominantHand.RIGHT,
        password="7nn6wNWBdh2X",
        uid=UUID("42e68894-30fb-11f0-9449-2dce6e7d0d6d"),
    )
    users.add(u1)

    file_mock = mocker.patch(
        "webapp.backend.repositories.users.open", mocker.mock_open(read_data=u1_json)
    )
    users.load()
    file_mock.assert_called_once_with(STORAGE_PATH, "rb")
    user_list = users.get_users()
    assert user_list == [u1]


def test_load_multiple_users(users: Users, mocker) -> None:
    u1_json = (
        b"[\n"
        + b"  {\n"
        + b'    "name": "USR1",\n'
        + b'    "arm_length": 45,\n'
        + b'    "shoulder_height": 145,\n'
        + b'    "dominant_hand": "R",\n'
        + b'    "password": "7nn6wNWBdh2X",\n'
        + b'    "uid": "42e68894-30fb-11f0-9449-2dce6e7d0d6d"\n'
        + b"  },\n"
        + b"  {\n"
        + b'    "name": "USR2",\n'
        + b'    "arm_length": 55,\n'
        + b'    "shoulder_height": 155,\n'
        + b'    "dominant_hand": "L",\n'
        + b'    "password": "7nn6wNWBdh2x",\n'
        + b'    "uid": "42e68894-30fb-11f0-9449-2dce6e7d0d6f"\n'
        + b"  }\n"
        + b"]\n"
    )
    u1: User = User(
        "USR1",
        45,
        145,
        DominantHand.RIGHT,
        password="7nn6wNWBdh2X",
        uid=UUID("42e68894-30fb-11f0-9449-2dce6e7d0d6d"),
    )
    users.add(u1)
    u2: User = User(
        "USR2",
        55,
        155,
        DominantHand.LEFT,
        password="7nn6wNWBdh2x",
        uid=UUID("42e68894-30fb-11f0-9449-2dce6e7d0d6f"),
    )

    file_mock = mocker.patch(
        "webapp.backend.repositories.users.open", mocker.mock_open(read_data=u1_json)
    )
    users.load()
    file_mock.assert_called_once_with(STORAGE_PATH, "rb")
    user_list = users.get_users()
    assert user_list == [u1, u2]
