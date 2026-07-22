from __future__ import annotations

import threading
import unittest
from urllib.request import urlopen

import launch_gui


class StandaloneLauncherTests(unittest.TestCase):
    def test_packaged_interface_exists(self) -> None:
        root = launch_gui.default_interface_root()
        self.assertTrue((root / "index.html").is_file())

    def test_server_exposes_the_interface_on_loopback(self) -> None:
        server = launch_gui.create_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            _, port = server.server_address[:2]
            with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                body = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn("Maintenance Impact Explorer", body)
            self.assertIn("Standalone architecture", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
