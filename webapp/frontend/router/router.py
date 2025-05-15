"""Implementation for JS based page router."""

from typing import Callable, Dict, Union

from nicegui import background_tasks, helpers, ui


class RouterFrame(ui.element, component="router_frame.js"):
    """Router frame intended to be replaced by JavaScript.

    NOTE: Used to prevent necessity of reloading the web-page.
    """

    pass


class Router:
    """Class handling the requests to reload the page."""

    def __init__(self) -> None:
        """Initialize Router instance."""
        self.routes: Dict[str, Callable] = {}
        self.content: ui.element | None = None

    def add(self, path: str) -> Callable:
        """Decorator to add path to router."""

        def decorator(func: Callable) -> Callable:
            self.routes[path] = func
            return func

        return decorator

    def open(self, target: Union[Callable, str]) -> None:
        """Handle page update requests."""
        if self.content is None:
            return

        if isinstance(target, str):
            path = target
            builder = self.routes[target]
        else:
            path = {v: k for k, v in self.routes.items()}[target]
            builder = target

        async def build() -> None:
            if self.content is None:
                return
            with self.content:
                ui.run_javascript(f"""
                    if (window.location.pathname !== "{path}") {{
                        history.pushState({{page: "{path}"}}, "", "{path}");
                    }}
                """)
                result = builder()
                if helpers.is_coroutine_function(builder):
                    await result

        self.content.clear()
        background_tasks.create(build())

    def frame(self) -> ui.element:
        """Get page content."""
        self.content = RouterFrame().on("open", lambda e: self.open(e.args))
        return self.content
