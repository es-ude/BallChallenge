#!/usr/bin/env python3

"""Entry point for the web application."""

import base64

import cv2
import numpy as np
from fastapi import Response
from nicegui import app, run, ui

from webapp.backend.config.settings import Configuration, load_yaml
from webapp.backend.repositories.notes import Nodes
from webapp.backend.repositories.users import Users
from webapp.frontend.pages.datacollect import DataCollectPage
from webapp.frontend.pages.user import user_page
from webapp.frontend.router.router import Router

user_repo: Users
nodes: Nodes
cv2_capture: cv2.VideoCapture


def setup_backend() -> None:
    """Setup Data Backend."""
    global user_repo, nodes, cv2_capture
    configuration: Configuration = load_yaml("settings/config.yaml")
    user_repo = Users(path=configuration.app.user_file)
    nodes = Nodes(
        domain=configuration.mqtt.domain,
        host=configuration.mqtt.host,
        port=configuration.mqtt.port,
        id=configuration.mqtt.id,
    )
    cv2_capture = cv2.VideoCapture(0)


def __convert(frame: np.ndarray) -> bytes:
    """Converts a frame from OpenCV to a JPEG image.

    This is a free function (not in a class or inner-function),
    to allow run.cpu_bound to pickle it and send it to a separate process.
    """
    _, imencode_image = cv2.imencode(".jpg", frame)
    return imencode_image.tobytes()


@ui.page("/video/frame")
async def grab_video_frame() -> Response:
    global cv2_capture
    PLACEHOLDER = Response(
        content=base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII=".encode(
                "ascii"
            )
        ),
        media_type="image/png",
    )
    if not cv2_capture.isOpened():
        return PLACEHOLDER
    _, frame = await run.io_bound(cv2_capture.read)
    if frame is None:
        return PLACEHOLDER
    jpeg = await run.cpu_bound(__convert, frame)
    return Response(content=jpeg, media_type="image/jpeg")


@ui.page("/")
@ui.page("/{_:path}")
def setup_ui() -> None:
    """Setup basic UI for the web-app."""
    global nodes, user_repo, cv2_capture
    router = Router()

    @router.add("/")
    def homepage() -> None:
        ui.label("Ball Challenge Web-App").classes("text-2xl fixed-center")

    @router.add("/user")
    def add_user() -> None:
        user_page(user_repo)

    @router.add("/collect")
    def collect_data() -> None:
        DataCollectPage(nodes=nodes, users=user_repo, video=cv2_capture)

    # navigation buttons to switch between the different pages
    with ui.row():
        ui.button("Home", on_click=lambda: router.open(homepage)).classes("min-w-32")
        ui.button("Add User", on_click=lambda: router.open(add_user)).classes(
            "min-w-32"
        )
        ui.button("Record Sample", on_click=lambda: router.open(collect_data)).classes(
            "min-w-32"
        )

    # this places the content which should be displayed
    with ui.row().classes("w-full h-full"):
        router.frame().classes("w-full h-full p-4")


def start_ui(host: str = "localhost", port: int = 8081) -> None:
    """Start nicegui UI."""
    app.on_startup(setup_backend)
    ui.run(host=host, port=port, reload=True, title="Ball Challenge")


start_ui()
