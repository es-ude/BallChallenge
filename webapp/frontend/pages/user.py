"""Frontend to add a User."""

from nicegui import ui
from qrcode import QRCode, constants
from qrcode.image.svg import SvgPathFillImage

from webapp.backend.entities.user import DominantHand, User
from webapp.backend.repositories.users import DatasetError, Users


def user_page(user_repo: Users) -> None:
    """Add new User."""

    def show_qr_code(uid: str, password: str) -> None:
        __qr = QRCode(
            version=None,
            error_correction=constants.ERROR_CORRECT_L,
            box_size=20,
            border=5,
        )
        __qr.make(fit=True)
        __qr.add_data(f"UID:{uid};PASSWORD:{password}")
        qr_code = __qr.make_image(
            image_factory=SvgPathFillImage, fill_color="black", back_color="white"
        )

        with ui.dialog() as dialog, ui.card().tight():
            with ui.row().classes("w-full justify-center"):
                ui.html(qr_code.to_string(encoding="unicode"))
            with ui.card_section():
                with ui.row(wrap=False):
                    ui.label(f"UID: {uid}").classes("text-2xl")
                with ui.row(wrap=False):
                    ui.label(f"Password: {password}").classes("text-2xl")
                with ui.row(wrap=False).classes("justify-center mt-8"):
                    ui.button("Done", on_click=dialog.close).props("w-100 text-middle")

        dialog.open()

    def add_new_user() -> None:
        new_user = User(
            name=user_name.value,
            arm_length=arm_length.value,
            shoulder_height=shoulder_height.value,
            dominant_hand=dominant_hand.value,
        )
        try:
            user_repo.insert(new_user)
            user_name.set_value(None)
            arm_length.set_value(None)
            shoulder_height.set_value(None)
            dominant_hand.set_value(None)
            show_qr_code(str(new_user.uid), new_user.password)
        except DatasetError as e:
            ui.notify(f"Error: {e}", type="warning")

    with ui.grid().classes("gap-4"):
        ui.label("Add USER").classes("text-2xl")
        user_name = ui.input(label="Nick Name")
        arm_length = ui.number(
            label="Arm Length", step=1, precision=-1, min=20, max=150, suffix="cm"
        )
        shoulder_height = ui.number(
            label="Shoulder Height",
            step=1,
            precision=-1,
            min=100,
            max=250,
            suffix="cm",
        )
        with ui.row(align_items="center"):
            ui.label("Dominant Hand:").classes("text-gray-600 text-end")
            dominant_hand = ui.radio({h.value: h.name for h in DominantHand}).props(
                "inline align-middle"
            )
        ui.button(
            text="save",
            on_click=add_new_user,
        )
