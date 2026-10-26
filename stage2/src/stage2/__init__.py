from stage2.ui import build_app


def main() -> None:
    build_app().launch(server_name="127.0.0.1", server_port=7860)
