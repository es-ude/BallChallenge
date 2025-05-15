"""Front-end for the data collect service."""

import time
from asyncio import sleep

import cv2
import numpy as np
from elasticai.protocol.data_requester import DataRequester, DeviceNotAvailableError
from matplotlib import pyplot as plt
from nicegui import ui

from webapp.backend.entities.user import User
from webapp.backend.repositories.notes import Nodes
from webapp.backend.repositories.users import Users


class DataCollectPage:
    """Data collection service."""

    __COUNTDOWN_COLORS: dict[int, str]
    __COUNTDOWN_VALUES: dict[int, str]

    # external data sources
    __users: Users
    __nodes: Nodes

    # current state
    __selected_user: dict[str, str]
    __selected_node: dict[str, str]

    def __init__(self, nodes: Nodes, users: Users, video: cv2.VideoCapture) -> None:
        """Initialize Webpage."""
        self.__COUNTDOWN_COLORS = {
            2: "bg-red",
            1: "bg-yellow",
            0: "bg-green",
        }
        self.__COUNTDOWN_VALUES = {
            0: "Throw",
        }
        self.__users = users
        self.__nodes = nodes
        self.__selected_user = {}
        self.__selected_node = {}

        with ui.grid(columns=3).classes("gap-4 items-center"):
            with ui.column().classes("col-span-2"):
                ui.select(
                    options=self.__users.find_all(),
                    label="User",
                    clearable=True,
                    with_input=True,
                ).classes("w-full").bind_value(self.__selected_user)
                ui.select(
                    options=self.__nodes.find_all(),
                    label="Node",
                    clearable=True,
                    with_input=True,
                ).classes("w-full").bind_value(self.__selected_node)
                ui.button("Start Recording", on_click=self.__record_sample).classes(
                    "w-full"
                )
            with ui.column():
                video_image = ui.interactive_image()
                ui.timer(
                    interval=0.1,
                    callback=lambda: video_image.set_source(
                        f"/video/frame?{time.time()}"
                    ),
                )

    def __validate_user(self, user_id: str | None) -> User | None:
        if user_id is None:
            print("NO USER SELECTED!")  # TODO: add warning
            return None
        if (user := self.__users.find_by_uuid(user_id)) is None:
            print("USER DOES NOT EXIST!")  # TODO: add warning
            return None
        return user

    def __validate_node(self, node: str | None) -> DataRequester | None:
        if node is None:
            print("NO NODE SELECTED!")  # TODO: add warning
            return None
        if (recorder := self.__nodes.find_by_name(node)) is None:
            print("NODE NOT AVAILABLE!")  # TODO: add warning
            return None
        return recorder

    async def __show_countdown(self) -> None:
        countdown: dict[str, str] = {}
        with (
            ui.dialog()
            .props('persistent backdrop-filter="blur(8px) brightness(40%)"')
            .classes("w-full h-full") as dialog,
            ui.card()
            .props("flat bordered")
            .classes("w-full h-full place-items-center justify-center") as card,
        ):
            dialog.open()
            ui.label().classes("text-9xl").bind_text_from(countdown, "value")
            for i in reversed(range(0, 6)):
                card.classes(
                    remove=" ".join(self.__COUNTDOWN_COLORS.values()),
                    add=self.__COUNTDOWN_COLORS.get(i, "bg-red"),
                )
                countdown["value"] = self.__COUNTDOWN_VALUES.get(i, str(i))
                await sleep(1)
            dialog.props(remove="persistent")
            await sleep(5)
            dialog.close()

    def __convert_recorded_data(self, data: str) -> np.ndarray:
        samples: list[str] = data.split(";")
        sample_array: list[np.ndarray] = [
            np.fromstring(string=sample, sep=",") for sample in samples
        ]
        return np.array(sample_array)

    def __show_results(self, samples: np.ndarray) -> None:
        with ui.grid(columns=3):
            with ui.row():
                # TODO: show recordings (graph)
                # TODO: show image with coordinates
                with ui.pyplot():
                    plt.plot(np.linspace(0, 5), samples[:, 0], "-")
                    plt.title("Data: X")
                with ui.pyplot():
                    plt.plot(np.linspace(0, 5), samples[:, 1], "-")
                    plt.title("Data: Y")
                with ui.pyplot():
                    plt.plot(np.linspace(0, 5), samples[:, 2], "-")
                    plt.title("Data: Z")
            with ui.row():
                # TODO: show prediction (heatmap?)
                pass

    async def __record_sample(self) -> None:
        if (user := self.__validate_user(self.__selected_user["value"])) is None:
            return
        if (node := self.__validate_node(self.__selected_node["value"])) is None:
            return

        try:
            node.start()
            await self.__show_countdown()
            raw_samples: str = data if (data := node.get_data()) is not None else ""
            node.stop()
            samples: np.ndarray = self.__convert_recorded_data(raw_samples)
            self.__show_results(samples)
            # TODO: store data
        except DeviceNotAvailableError:
            print("Connection failed!")  # TODO: add warning
