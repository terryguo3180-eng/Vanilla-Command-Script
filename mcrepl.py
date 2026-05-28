from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import select
import shutil
import socket
import ssl
import struct
import subprocess
import tempfile
import time
import urllib.request

if platform.system() != "Windows":
    import signal

from vcs.ansi import AnsiCode
from vcs.repl import REPL


def color_text(text: str, color: str = AnsiCode.RESET, bold: bool = False) -> str:
    return (AnsiCode.BOLD if bold else "") + color + text + AnsiCode.RESET

def print_error(text: str, end="\n", flush=False) -> None:
    print(color_text(text, AnsiCode.RED, bold=True), end=end, flush=flush)

def print_success(text: str, end="\n", flush=False) -> None:
    print(color_text(text, AnsiCode.GREEN), end=end, flush=flush)

def print_info(text: str, end="\n", flush=False) -> None:
    print(text, end=end, flush=flush)

def print_warning(text: str, end="\n", flush=False) -> None:
    print(color_text(text, AnsiCode.YELLOW), end=end, flush=flush)


RCON_HOST = "127.0.0.1"
RCON_PORT = 25575
RCON_PASS = "123456"

class MCRconException(Exception): ...

class MCRcon:
    # Code adapted from barneygale/MCRcon
    sock = None

    def __init__(self, host, password, port=25575, tlsmode=0, timeout=5):
        self.host = host
        self.password = password
        self.port = port
        self.tlsmode = tlsmode
        self.timeout = timeout

        def timeout_handler(signum, frame):
            raise MCRconException("Connection timeout error")
        
        if platform.system() != "Windows":
            signal.signal(signal.SIGALRM, timeout_handler)  # type: ignore

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, type, value, tb):
        self.disconnect()

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        if self.tlsmode > 0:
            ctx = ssl.create_default_context()
            if self.tlsmode > 1:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            self.sock = ctx.wrap_socket(self.sock, server_hostname=self.host)

        self.sock.connect((self.host, self.port))
        self._send(3, self.password)

    def disconnect(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _read(self, length):
        if platform.system() != "Windows":
            signal.alarm(self.timeout)  # type: ignore
        data = b""
        while len(data) < length:
            assert self.sock is not None
            data += self.sock.recv(length - len(data))
        if platform.system() != "Windows":
            signal.alarm(0)  # type: ignore
        return data

    def _send(self, out_type, out_data):
        if self.sock is None:
            raise MCRconException("Must connect before sending data")

        out_payload = (
            struct.pack("<ii", 0, out_type) + out_data.encode("utf8") + b"\x00\x00"
        )
        out_length = struct.pack("<i", len(out_payload))
        self.sock.send(out_length + out_payload)

        in_data = ""
        while True:
            (in_length,) = struct.unpack("<i", self._read(4))
            in_payload = self._read(in_length)
            in_id, in_type = struct.unpack("<ii", in_payload[:8])
            in_data_partial, in_padding = in_payload[8:-2], in_payload[-2:]

            if in_padding != b"\x00\x00":
                raise MCRconException("Incorrect padding")
            if in_id == -1:
                raise MCRconException("Login failed")

            in_data += in_data_partial.decode("utf8")

            if len(select.select([self.sock], [], [], 0)[0]) == 0:
                return in_data

    def command(self, command):
        result = self._send(2, command)
        time.sleep(0.003)  # MC-72390 workaround
        return result


def get_latest_paper_jar():
    print_info("Fetching latest Paper version...")
    
    ver_url = "https://api.papermc.io/v2/projects/paper"
    try:
        with urllib.request.urlopen(ver_url) as r:
            info = json.loads(r.read())
        vers = info["versions"]
        latest_ver = vers[-1]
        print_info(f"Latest Paper game version: {latest_ver}")
        
        builds_url = f"https://api.papermc.io/v2/projects/paper/versions/{latest_ver}/builds"
        with urllib.request.urlopen(builds_url) as r:
            builds_info = json.loads(r.read())
        if not builds_info["builds"]:
            raise Exception(f"No builds available for version {latest_ver}")
        latest = builds_info["builds"][0]
        buildno = latest["build"]
        filename = latest["downloads"]["application"]["name"]
        
        download_url = (
            f"https://api.papermc.io/v2/projects/paper/versions/"
            f"{latest_ver}/builds/{buildno}/downloads/{filename}"
        )
        print_info(f"Download URL: {download_url}")
        
        print_info("Downloading Paper... (this may take a moment)")
        urllib.request.urlretrieve(download_url, "server.jar")
        print_success("Paper download complete")
        
        f_size = Path("server.jar").stat().st_size
        if f_size < 10 * 1024 * 1024:
            raise Exception(f"Downloaded file too small ({f_size} bytes)")
        return True
        
    except Exception as e:
        print_error(f"Failed to download Paper: {e}")
        print_warning("You can manually download Paper from https://papermc.io/downloads")
        print_warning("and place it as 'server.jar' in the current directory.")
        return False


def wait_for_rcon(host, port, timeout=45):
    print_info("Waiting for server to start...", end="", flush=True)
    start = time.time()
    dots = 0
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((host, port))
            sock.close()
            print()  # newline after dots
            return True
        except:
            print_info(".", end="", flush=True)
            dots += 1
            time.sleep(1)
    print()  # final newline
    return False


def send_cmd(cmd, host=RCON_HOST, port=RCON_PORT, pwd=RCON_PASS):
    try:
        with MCRcon(host, pwd, port=port) as mcr:
            resp = mcr.command(cmd)
            return resp
    except MCRconException as e:
        return color_text(f"RCON Error: {e}", AnsiCode.RED, bold=True)


def mc_to_ansi(text: str):
    color_map = {
        '0': '\033[30m', '1': '\033[34m', '2': '\033[32m', '3': '\033[36m',
        '4': '\033[31m', '5': '\033[35m', '6': '\033[33m', '7': '\033[37m',
        '8': '\033[90m', '9': '\033[94m', 'a': '\033[92m', 'b': '\033[96m',
        'c': '\033[91m', 'd': '\033[95m', 'e': '\033[93m', 'f': '\033[97m',
        'l': '\033[01m', 'o': '\033[03m', 'n': '\033[04m', 'm': '\033[09m',
        'r': '\033[00m', 'k': '',
    }
    
    pattern = re.compile(r'§([0-9a-fk-or])', re.IGNORECASE)
    result = pattern.sub(lambda m: color_map.get(m.group(1).lower(), ''), text)
    
    if result and not result.endswith('\033[00m'):
        result += '\033[00m'
    return result


class ServerLogFile:
    def __init__(self, path: str):
        self.path = Path(path)
        self.modified = self.get_last_modified()
        self.last_size = self.path.stat().st_size

    def get_last_modified(self):
        return time.ctime(self.path.stat().st_mtime)
        
    def was_updated(self):
        t = self.get_last_modified()
        if t > self.modified:
            self.modified = t
            return True
        else:
            return False

    def get_text(self, new_text=False):
        text = ""
        while len(text) == 0:
            with open(self.path, 'r') as f:
                if new_text:
                    f.seek(self.last_size)
                text = f.read()
            
            time.sleep(0.1)
            
        self.last_size = self.path.stat().st_size
        return text

    def find_reload_errors(self):
        text = self.get_text(True)

        lines = text.splitlines()
        errors = []
        detected = False

        for line in lines:
            if re.fullmatch(r'\[\d\d:\d\d:\d\d ERROR\]: Failed to load function \w+:\w+', line):
                errors.append(line)
                detected = True
            elif detected and line.startswith('['):
                detected = False
            elif detected and not line.startswith('\t') and not line.startswith('Caused by: '):
                errors.append(line)
        return errors


def copy_datapack(dpacks_dir: Path, dp_path: Path):
    if dp_path.is_file():
        dest = dpacks_dir / dp_path.name
        shutil.copy2(dp_path, dest)
        return color_text(f"Copied datapack file: {dp_path.name}", AnsiCode.GREEN)
    if dp_path.is_dir():
        dest = dpacks_dir / dp_path.name
        shutil.copytree(dp_path, dest)
        return color_text(f"Copied datapack folder: {dp_path.name}", AnsiCode.GREEN)
    return color_text(f"Failed to copy datapack: {dp_path.name}", AnsiCode.RED)


def main():
    argparser = argparse.ArgumentParser(description="Minecraft datapack testing tool")
    argparser.add_argument("datapack", help="path to datapack (folder or zip file)")
    args = argparser.parse_args()
    
    dp_path = Path(args.datapack).resolve()

    if not dp_path.exists():
        print_error(f"Error: {dp_path} does not exist")
        exit(1)
    
    test_dir = Path(tempfile.mkdtemp(prefix="test-"))
    print_info(f"Test directory: {test_dir}")
    
    sv_proc = None
    
    try:
        os.chdir(test_dir)
        
        if not Path("server.jar").exists():
            if not get_latest_paper_jar():
                exit(1)
        
        with open("eula.txt", "w") as f:
            f.write("eula=true")
        
        with open("server.properties", "w") as f:
            f.write("\n".join([
                "enable-rcon=true",
                f"rcon.port={RCON_PORT}",
                f"rcon.password={RCON_PASS}",
                "level-name=test",
                "max-tick-time=60000",
            ]))
        
        world_dir = Path("test")
        world_dir.mkdir(exist_ok=True)
        dpacks_dir = world_dir / "datapacks"
        dpacks_dir.mkdir(exist_ok=True)
        
        print(copy_datapack(dpacks_dir, dp_path))
        
        with open("server.log", "w") as f:
            sv_proc = subprocess.Popen(
                ["java", "-Xmx1G", "-jar", "server.jar", "nogui"],
                stdout=f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True
            )
        
        logfile = ServerLogFile("server.log")

        if not wait_for_rcon(RCON_HOST, RCON_PORT, timeout=45):
            print_error("\nServer startup timeout")
            print_warning("Last 10 lines of server.log:")
            lines = logfile.get_text().splitlines()
            for line in lines[-10:]:
                print(f"  {AnsiCode.DIM}{line.strip()}{AnsiCode.RESET}")
            exit(1)
        
        time.sleep(2)
        print_success("Server is ready!")
        logfile.get_text()

        def cli(prompt: str, line: str):
            if not line.startswith("/"):
                print(f"  {AnsiCode.DIM}{line}{AnsiCode.RESET}")
                return

            if line == "/stop":
                print_warning("  Type Ctrl+C to exit the program")
                return

            if line == "/reload":
                print("  " + copy_datapack(dpacks_dir, dp_path))

            resp = send_cmd(line)
            if resp:
                colored_resp = mc_to_ansi(resp.rstrip())
                if colored_resp.strip():
                    for resp_line in colored_resp.splitlines():
                        print(f"  {resp_line}")
            else:
                print(f"  {AnsiCode.DIM}Command executed with no output{AnsiCode.RESET}")

            if line == "/reload":
                errors = logfile.find_reload_errors()
                if errors:
                    print_warning("  Error detected in Minecraft log file:")
                    for error in errors:
                        print(f"    {AnsiCode.BG_RED}{error}{AnsiCode.RESET}")

        print()
        intro = color_text(
            "Minecraft REPL (Read-Eval-Print Loop) module "
            "for datapack testing, type Ctrl+C to exit the program",
            AnsiCode.MAGENTA, bold=True
        )
        REPL(cli, False).cmdloop(intro)
        
    except KeyboardInterrupt:
        print_warning("\nInterrupted")
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print_info("\nCleaning up temporary files...")
        if sv_proc:
            try:
                send_cmd("/stop")
                sv_proc.wait(timeout=10)
                print_success("Server stopped gracefully")
            except:
                sv_proc.terminate()
                sv_proc.wait(timeout=5)
                print_warning("Server was force-terminated")
        
        os.chdir(Path.home())
        try:
            shutil.rmtree(test_dir, ignore_errors=True)
            print_success(f"Removed test directory: {test_dir}")
        except Exception as e:
            print_warning(f"Could not fully remove {test_dir}: {e}")
        print_success("Done!")


if __name__ == "__main__":
    main()