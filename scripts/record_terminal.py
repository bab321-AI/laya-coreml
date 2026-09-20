"""Capture a real terminal session without streaming ANSI frames into a build log.

Usage: python scripts/record_terminal.py --log artifacts/session.ansi -- command ...
The child receives a 112-column, 38-row truecolor pseudo-terminal.
"""

import argparse
import errno
import fcntl
import os
import pty
import struct
import subprocess
import termios
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("xb") as output:
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 38, 112, 0, 0))
        child = subprocess.Popen(
            command,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env={**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"},
        )
        os.close(slave)
        try:
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                output.write(chunk)
        except BaseException:
            if child.poll() is None:
                child.terminate()
            child.wait()
            raise
        finally:
            os.close(master)
        return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
