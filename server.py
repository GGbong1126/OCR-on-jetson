# -*- coding: utf-8 -*-
"""干部人事档案 OCR 自动著录 Flask HTTP 服务。

对外接口（遵循《OCR服务能力接入规范 V1.1》）：
  POST /OCR      — Base64 单图 OCR，返回文本与原图像素坐标
  POST /Extract  — 基于 OCR 结果和动态字段定义提取结构化字段
"""
import os, sys, json, time, uuid, base64, tempfile, logging, argparse, shutil, subprocess, random, hashlib, hmac, math, re, binascii, secrets
from hashlib import sha256
from datetime import datetime
os.environ['DISABLE_AUTO_LOGGING_CONFIG'] = '1'

# ── 日志配置 ──
LOG_FILE = os.environ.get("AUTOOCR_LOG_FILE", os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "server.log"))
MAX_LOG_LINES = 5000

class RotatingFileHandler(logging.Handler):
    """循环覆盖日志处理器，超过 max_lines 行时保留最新内容"""
    def __init__(self, filepath, max_lines=MAX_LOG_LINES):
        super().__init__()
        self.filepath = filepath
        self.max_lines = max_lines
        self._lines_count = 0
        # 初始化时检查现有行数
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                self._lines_count = sum(1 for _ in f)

    def emit(self, record):
        try:
            msg = self.format(record)
            if self._lines_count >= self.max_lines:
                # 超过上限，读取后半部分保留
                keep = self.max_lines // 2
                lines = []
                if os.path.exists(self.filepath):
                    with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines[-keep:])
                self._lines_count = keep
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
            self._lines_count += 1
        except Exception:
            pass

# 配置 logger
logger = logging.getLogger('autoocr')
logger.setLevel(logging.INFO)
_handler = RotatingFileHandler(LOG_FILE, MAX_LOG_LINES)
_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(_handler)

def log(msg):
    """统一日志接口"""
    logger.info(msg)
    print(msg, flush=True)

from pathlib import Path

import platform
try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog, scrolledtext
except ImportError:
    # Headless Linux uses the explicit deployment activation CLI; GUI paths stay unchanged.
    tk = messagebox = simpledialog = scrolledtext = None

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

if platform.system() == "Windows":
    try:
        import winreg
    except ImportError:
        winreg = None

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from flask import Flask, request, jsonify, Response

# ── 本地模块 ──
import server_config as cfg
import windows_hardware
from ner_postprocess import predict_fields_ner, VALID_LABELS
from layout.layout_fusion import LayoutFusion, LayoutFusionConfig

from words import *




# ==================================================
# 基本配置
# ==================================================

# 授权重试与许可证配置（上一版清理重复代码时误删，必须保留）
RETRY_INTERVAL = 10
LICENSE_FILE = os.environ.get("AUTOOCR_LICENSE_FILE", "./license.dat")
APP_SEED = None  # Private legacy license material is excluded from this source export.

CHARSET = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'
BASE = len(CHARSET)               # 58
AUTH_CODE_LEN = 32                # 授权码固定长度
MAX_NUM = BASE ** AUTH_CODE_LEN   # 最大整数空间

SECRET_KEY = None  # Private legacy license material is excluded from this source export.
password3 = None  # Private legacy license material is excluded from this source export.
key29 = None  # Private legacy license material is excluded from this source export.
COUNTER_INDICES = [22, 21, 24, 29, 15, 1, 16]
INITIAL_COUNTER = 20000

# 明文布局（字节级）
# [1] sn_len
# [1] pos
# [19] payload（包含硬盘序列号）
PLAIN_BYTES_LEN = 21
PAYLOAD_LEN = 19


# ==================================================
# 硬盘序列号获取
# ==================================================



def _require_private_license_config():
    """Public source excludes private credentials; legacy operations fail closed."""
    if any(value is None for value in (APP_SEED, SECRET_KEY, password3, key29)):
        raise RuntimeError("Private legacy license configuration is not included in this source export")


def _get_disk_serial() -> str:
    if sys.platform.startswith('win'):
        out = windows_hardware.check_output(
            'wmic diskdrive get SerialNumber',
            shell=True,
            text=True
        )
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if len(lines) < 2:
            raise RuntimeError("未找到硬盘序列号")
        return lines[1]

    elif sys.platform.startswith('linux'):
        sys_block = Path("/sys/block")
        # 过滤：仅保留包含 device 目录的设备（自动排除 loop、ramdisk 等纯虚拟设备）
        disks = [d for d in sys_block.iterdir() if (d / "device").exists()]

        def sort_key(d):
            if d.name.startswith("sd"):
                return 0
            if d.name.startswith("nvme"):
                return 1
            return 2

        for d in sorted(disks, key=sort_key):
            # ---------------- 策略 1: 读取 /sys 虚拟文件系统 ----------------
            sys_serial_path = d / "device" / "serial"
            try:
                if sys_serial_path.exists():
                    sn = sys_serial_path.read_text(encoding='utf-8').strip()
                    # 过滤硬件/驱动返回的无效占位符
                    if sn and sn.upper() not in ('NONE', 'UNKNOWN', 'NULL', ''):
                        return sn
            except (PermissionError, OSError):
                # 权限不足或 sysfs 读取异常，静默进入 fallback
                pass

            # ---------------- 策略 2: lsblk 命令 fallback ----------------
            try:
                res = subprocess.run(
                    ['lsblk', '-ndo', 'SERIAL', f'/dev/{d.name}'],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5
                )
                sn = res.stdout.strip()
                if sn and sn.upper() not in ('NONE', 'UNKNOWN', 'NULL', ''):
                    return sn
            except (subprocess.SubprocessError, FileNotFoundError, TimeoutError):
                # lsblk 未安装、执行失败或超时，跳过当前设备继续尝试
                continue

        raise RuntimeError("无法读取硬盘序列号")

    else:
        raise RuntimeError(f"不支持的平台: {sys.platform}")

def verify_disk_serial(code_disk_serial, current_disk_serial):
    """
    验证授权码中的硬盘序列号是否与当前设备匹配

    参数:
        code_disk_serial: 从授权码解析出的硬盘序列号
        current_disk_serial: 当前设备的硬盘序列号
    
    返回:
        bool: 是否匹配
    """
    # 将当前硬盘序列号按照与编码时相同的规则处理
    if len(current_disk_serial) < 17:
        normalized_current = current_disk_serial + '%' * (17 - len(current_disk_serial))
    else:
        normalized_current = current_disk_serial[:17]
    
    # 去除填充字符（与decode_all中的处理一致）
    normalized_current = normalized_current.rstrip('%')
    
    # 比较
    return normalized_current == code_disk_serial

# ==================================================
# 明文构造
# ==================================================

def build_plain_bytes() -> bytes:
    disk_sn = _get_disk_serial()
    sn_len = len(disk_sn)

    if sn_len > PAYLOAD_LEN:
         disk_sn = disk_sn[-16:]

    sn_len = len(disk_sn)
    pos = random.randint(0, PAYLOAD_LEN - sn_len)

    payload = bytearray(PAYLOAD_LEN)
    for i in range(PAYLOAD_LEN):
        if pos <= i < pos + sn_len:
            payload[i] = ord(disk_sn[i - pos])
        else:
            payload[i] = ord(random.choice(CHARSET))

    return bytes([sn_len, pos]) + payload  # 共 21 字节


# ==================================================
# 整数级可逆混淆（核心）
# ==================================================

def _mask() -> int:
    # 取 24 字节掩码（192 bit < Base58^32）
    _require_private_license_config()
    return int.from_bytes(sha256(SECRET_KEY).digest()[:24], 'big')

def mix_number(num: int) -> int:
    return (num + _mask()) % MAX_NUM

def unmix_number(num: int) -> int:
    return (num - _mask()) % MAX_NUM


# ==================================================
# Base58 固定长度编码 / 解码
# ==================================================

def int_to_fixed_base58(num: int) -> str:
    chars = []
    for _ in range(AUTH_CODE_LEN):
        num, rem = divmod(num, BASE)
        chars.append(CHARSET[rem])
    return ''.join(reversed(chars))

def fixed_base58_to_int(code: str) -> int:
    num = 0
    for c in code:
        num = num * BASE + CHARSET.index(c)
    return num


# ==================================================
# 授权码生成
# ==================================================

def generate_auth_code() -> str:
    plain = build_plain_bytes()                  # 21 bytes
    num = int.from_bytes(plain, 'big')
    num = mix_number(num)
    return int_to_fixed_base58(num)

def extract_serial_from_auth(auth_code: str) -> str:
    if len(auth_code) != AUTH_CODE_LEN:
        return ""
    try:
        num = fixed_base58_to_int(auth_code)
        plain = unmix_number(num).to_bytes(PLAIN_BYTES_LEN, 'big')
        sn_len = plain[0]
        pos = plain[1]
        # payload 从索引2开始，直接按记录的长度和偏移截取
        return plain[2 + pos: 2 + pos + sn_len].decode('utf-8', errors='ignore')
    except Exception:
        return ""

# 对应服务端 encrypt_auth_code 的逆操作
# ================= 替换原有的 decrypt_activation_code =================
def base58_encode_fixed(data: bytes, length: int = AUTH_CODE_LEN) -> str:
    num = int.from_bytes(data, 'big')

    chars = []
    for _ in range(length):
        num, rem = divmod(num, BASE)
        chars.append(CHARSET[rem])

    return ''.join(reversed(chars))

# encrypt_auth_code 相关常量和函数
_positions1 = [12, 5, 28, 1, 17, 23, 0, 8, 6, 30, 3, 19, 7, 15, 2, 9, 14, 11, 4, 21, 10, 13, 29, 27, 20, 16, 22, 18, 31, 24, 25, 26]
_positions2 = [3, 21, 7, 30, 2, 18, 9, 26, 12, 14, 0, 28, 6, 17, 25, 19, 1, 24, 5, 31, 22, 10, 13, 27, 20, 11, 29, 8, 15, 4, 16, 23]

def encrypt_auth_code(auth_code: str) -> str:
    if len(auth_code) != 32:
        raise ValueError("授权码必须为32位")
    # 第一轮位移
    shifted1 = []
    for i, c in enumerate(auth_code):
        idx = CHARSET.index(c)
        shift_val = (i + 5) * 13
        new_idx = (idx + shift_val) % len(CHARSET)
        shifted1.append(CHARSET[new_idx])
    # 第一轮换位
    rearranged1 = [''] * 32
    for i, pos in enumerate(_positions1):
        rearranged1[i] = shifted1[pos]
    # 第二轮位移
    shifted2 = []
    for i, c in enumerate(rearranged1):
        idx = CHARSET.index(c)
        delta = ((i ^ 7) + 11) * 9
        new_idx = (idx + delta) % len(CHARSET)
        shifted2.append(CHARSET[new_idx])
    # 第二轮换位
    rearranged2 = [''] * 32
    for i, pos in enumerate(_positions2):
        rearranged2[i] = shifted2[pos]
    return ''.join(rearranged2)

def verify_activation_code(auth_code: str, activation_code: str) -> bool:

    if len(auth_code) != AUTH_CODE_LEN:
        return False

    if len(activation_code) != AUTH_CODE_LEN:
        return False

    if any(c not in CHARSET for c in auth_code + activation_code):
        return False

    try:
        serial = extract_serial_from_auth(auth_code)
        if not serial:
            return False

        mac_bytes = hmac.new(
            serial.encode(),
            auth_code.encode(),
            hashlib.sha256
        ).digest()[:20]

        expected_code = base58_encode_fixed(mac_bytes)

        #再进行混淆加密
        expected_code = encrypt_auth_code(expected_code)

        return hmac.compare_digest(expected_code, activation_code)

    except Exception:
        return False



# ====================================================================

def get_device_fingerprint():
    info = []
    # CPU 信息
    try:
        cpu_info = platform.processor()
        if not cpu_info:
            # 有些系统返回空，用 wmic 获取
            result = windows_hardware.check_output("wmic cpu get Name", shell=True)
            lines = result.decode().splitlines()
            cpu_info = [line.strip() for line in lines if line.strip() and "Name" not in line][0]
        info.append(cpu_info)
    except Exception:
        info.append("cpu_error")

    # 操作系统信息（名称 + 版本 + 架构）
    try:
        system_info = f"{platform.system()}_{platform.version()}_{platform.machine()}"
        info.append(system_info)
    except Exception:
        info.append("system_error")

    # 主板信息（Manufacturer + Product）
    try:
        result = windows_hardware.check_output("wmic baseboard get Manufacturer,Product", shell=True)
        lines = result.decode().splitlines()
        board_info = [line.strip() for line in lines if line.strip() and "Manufacturer" not in line]
        motherboard = board_info[0] if board_info else "board_none"
        info.append(motherboard)
    except Exception:
        info.append("board_error")

    # 硬盘序列号
    try:
        result = windows_hardware.check_output("wmic diskdrive get SerialNumber", shell=True)
        lines = result.decode().splitlines()
        serials = [line.strip() for line in lines if line.strip() and "SerialNumber" not in line]
        disk_serial = serials[0] if serials else "disk_none"
        info.append(disk_serial)
    except Exception:
        info.append("disk_error")
    return "_".join(info)

def get_device_hash():
    _require_private_license_config()
    base_str = get_device_fingerprint()+APP_SEED
    # print(base_str)
    hash_val = hashlib.sha256(base_str.encode()).hexdigest()
    return hash_val

def create_license_file(auth_code, fingerprint):
    with open(LICENSE_FILE, "w") as f:
        json.dump({
            "auth_code": auth_code,
            "fingerprint": fingerprint
        }, f)

def validate_license():
    try:
        with open(LICENSE_FILE, "r") as f:
            data = json.load(f)
            # print(data["fingerprint"])
            # print(get_device_hash())
        return data["fingerprint"] == get_device_hash()
    except:
        return False

def show_auth_code_popup(code):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(name_warning, f"{name_please_send_license_code_to_admin}\n\n{code}")
    root.destroy()

def prompt_for_activation_code():
    root = tk.Tk()
    root.withdraw()
    code = simpledialog.askstring(name_input_activation_code, name_input_activation_code)
    root.destroy()
    return code.strip() if code else ""

def auth_activation_dialog(auth_code):
    """
    弹出一个窗口，左侧显示授权码，右侧输入激活码，
    用户点击确认后返回输入的激活码，点击退出返回空字符串。
    """
    result = {'activation_code': None}

    def on_confirm():
        code = entry_activation.get().strip()
        if len(code) != AUTH_CODE_LEN :
            messagebox.showerror(name_error, name_activation_length_error)
            return
        result['activation_code'] = code
        dialog.destroy()

    def on_exit():
        result['activation_code'] = ''
        dialog.destroy()

    dialog = tk.Tk()
    dialog.title(name_device_license)

    # 授权码标签
    label_auth = tk.Label(dialog, text=f"{name_please_send_license_code_to_admin}\n\t{auth_code}", justify='left')
    label_auth.grid(row=0, column=0, padx=32, pady=32)

    # 激活码输入框
    frame_right = tk.Frame(dialog)
    frame_right.grid(row=0, column=1, padx=32, pady=32)

    label_input = tk.Label(frame_right, text=name_input_activation_code)
    label_input.pack(pady=(0,5))

    entry_activation = tk.Entry(frame_right, width=15, font=('Consolas', 14))
    entry_activation.pack()

    # 按钮
    btn_confirm = tk.Button(frame_right, text=name_yes, command=on_confirm)
    btn_confirm.pack(pady=5, fill='x')

    btn_exit = tk.Button(frame_right, text=name_exit, command=on_exit)
    btn_exit.pack(fill='x')

    dialog.mainloop()
    return result['activation_code']

def show_device_code_popup(encrypted_device_id):
    def copy_to_clipboard():
        root.clipboard_clear()
        root.clipboard_append(encrypted_device_id.decode())
        root.update()  # 确保剪贴板更新
        root.destroy()  # 关闭窗口

    # 创建主窗口
    root = tk.Tk()
    root.title(name_tips)
    root.geometry("400x200")

    # 创建一个标签
    label = tk.Label(root, text=name_please_send_license_code_to_admin)
    label.pack(pady=32)

    # 创建一个可滚动的文本框
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=5)
    text_area.insert(tk.INSERT, encrypted_device_id.decode())
    text_area.pack(pady=32)

    # 使文本框中的内容可复制
    text_area.configure(state='normal')
    text_area.tag_add(tk.SEL, "1.0", tk.END)
    text_area.mark_set(tk.INSERT, "1.0")
    text_area.configure(state='disabled')

    # 运行主循环
    root.mainloop()


# ==================================================
# 计数授权（迁移自 app1.1.11_511_2.py）
# ==================================================

def get_hidden_app_dir():
    system_name = platform.system()
    if system_name == "Windows":
        base_dir = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        hidden_path = os.path.join(base_dir, 'Microsoft', 'Windows', 'Explorer')
    else:
        hidden_path = os.path.join(os.path.expanduser('~'), '.cache', 'sys_config_d')
    os.makedirs(hidden_path, exist_ok=True)
    if system_name == "Windows":
        try:
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(hidden_path, 0x02)
        except Exception:
            pass
    return hidden_path


def get_new_license_path():
    """获取隐藏目录中的余额文件路径。"""
    return os.path.join(get_hidden_app_dir(), 'sys_kernel_event.log')


def get_new_keys_path():
    """获取隐藏目录中的已使用充值码记录路径。"""
    return os.path.join(get_hidden_app_dir(), 'IconCache_ver.db')


def migrate_old_data():
    """将 App 旧版本的余额和充值码记录迁移到当前隐藏目录。"""
    system_path = os.path.expanduser('~')
    if platform.system() == "Windows":
        old_dir = os.path.join(system_path, 'Documents', 'Winlib')
        old_license_name = 'winlicense.log'
    else:
        old_dir = os.path.join(system_path, 'Documents', 'Syslib')
        old_license_name = 'syslicense.log'

    old_license_path = os.path.join(old_dir, old_license_name)
    old_keys_path = os.path.join(old_dir, 'sys_ver_control.dat')
    new_license_path = get_new_license_path()
    new_keys_path = get_new_keys_path()

    if os.path.exists(old_keys_path) and not os.path.exists(new_keys_path):
        try:
            print("Migrating system keys...", flush=True)
            shutil.move(old_keys_path, new_keys_path)
        except Exception as exc:
            print(f"Migration error (keys): {exc}", flush=True)

    if os.path.exists(old_license_path) and not os.path.exists(new_license_path):
        try:
            print("Migrating system logs...", flush=True)
            shutil.move(old_license_path, new_license_path)
        except Exception as exc:
            print(f"Migration error (logs): {exc}", flush=True)

    try:
        if os.path.exists(old_dir) and not os.listdir(old_dir):
            os.rmdir(old_dir)
    except Exception:
        pass


def get_license_file_path():
    return get_new_license_path()


def aes_encrypt_long(text):
    """使用与 App 相同的 AES-256-CBC 格式加密已用码记录。"""
    _require_private_license_config()
    key = hashlib.sha256(password3.encode()).digest()
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted_bytes = cipher.encrypt(pad(text.encode('utf-8'), AES.block_size))
    return base64.b64encode(iv + encrypted_bytes).decode('utf-8')


def aes_decrypt_long(encrypted_text):
    """解密 App 格式的数据；格式错误时返回 None。"""
    _require_private_license_config()
    try:
        if not encrypted_text:
            return None
        data = base64.b64decode(encrypted_text)
        iv = data[:16]
        ciphertext = data[16:]
        key = hashlib.sha256(password3.encode()).digest()
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_bytes = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted_bytes.decode('utf-8')
    except Exception:
        return None


def get_used_keys_path():
    return get_new_keys_path()


def get_linux_backup_path():
    backup_dir = os.path.join(os.path.expanduser('~'), '.local', 'share', '.python_sys_cache')
    os.makedirs(backup_dir, exist_ok=True)
    return os.path.join(backup_dir, 'sys_config.db')


def get_backup_data():
    """从 Windows 注册表或类 Unix 隐藏文件读取充值码备份。"""
    if platform.system() == "Windows":
        if winreg is None:
            return []
        reg_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SysVer"
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_READ)
            try:
                value, _ = winreg.QueryValueEx(key, "ConfigData")
                decrypted = aes_decrypt_long(value)
                return json.loads(decrypted) if decrypted else []
            except FileNotFoundError:
                return []
            finally:
                winreg.CloseKey(key)
        except Exception:
            return []

    backup_path = get_linux_backup_path()
    if not os.path.exists(backup_path):
        return []
    try:
        with open(backup_path, 'r', encoding='utf-8') as file:
            decrypted = aes_decrypt_long(file.read().strip())
            return json.loads(decrypted) if decrypted else []
    except Exception:
        return []


def save_backup_data(hash_list):
    """将已用码哈希写入第二份隐藏备份。"""
    try:
        encrypted_content = aes_encrypt_long(json.dumps(hash_list))
        if platform.system() == "Windows":
            if winreg is None:
                raise RuntimeError("winreg is unavailable")
            reg_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SysVer"
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, reg_path)
            try:
                winreg.SetValueEx(key, "ConfigData", 0, winreg.REG_SZ, encrypted_content)
            finally:
                winreg.CloseKey(key)
        else:
            with open(get_linux_backup_path(), 'w', encoding='utf-8') as file:
                file.write(encrypted_content)
    except Exception as exc:
        print(f"Backup Warning: {exc}", flush=True)


def save_used_hashes_internal(hash_list, write_backup=True):
    file_path = get_used_keys_path()
    try:
        encrypted_content = aes_encrypt_long(json.dumps(hash_list))
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(encrypted_content)
    except Exception as exc:
        print(f"File Save Error: {exc}", flush=True)
        raise

    if write_backup:
        save_backup_data(hash_list)


def save_used_hashes(hash_list):
    save_used_hashes_internal(hash_list, write_backup=True)


def load_used_hashes():
    """合并主记录和备份；任一副本丢失或滞后时自动修复。"""
    file_path = get_used_keys_path()
    file_hashes = None
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    decrypted = aes_decrypt_long(content)
                    if decrypted:
                        file_hashes = json.loads(decrypted)
        except Exception:
            file_hashes = None

    backup_hashes = get_backup_data()
    if file_hashes is None and not backup_hashes:
        return []
    if file_hashes is None and backup_hashes:
        print(">>> [Security] 检测到主记录丢失，正在从备份恢复...", flush=True)
        save_used_hashes_internal(backup_hashes, write_backup=False)
        return backup_hashes
    if file_hashes is not None and backup_hashes:
        merged = list(set(file_hashes) | set(backup_hashes))
        if len(merged) > len(file_hashes) or len(merged) > len(backup_hashes):
            save_used_hashes(merged)
        return merged
    return file_hashes if file_hashes else []


MIGRATION_SIGNATURE = "##SECURE_BINDING_V1##"


def check_integrity_and_usability(activation_code):
    """校验安全记录完整性，并拒绝重复使用同一充值码。"""
    used_keys_path = get_used_keys_path()
    license_path = get_license_file_path()
    has_log = os.path.exists(license_path)
    has_keys = os.path.exists(used_keys_path)

    if has_log and not has_keys:
        is_missing_after_migration = False
        try:
            with open(license_path, 'r', encoding='utf-8') as file:
                is_missing_after_migration = MIGRATION_SIGNATURE in file.read().strip()
        except Exception:
            pass

        backup_data = get_backup_data()
        if backup_data:
            print(">>> [Security] 文件丢失但发现备份，执行强制恢复。", flush=True)
            save_used_hashes(backup_data)
        elif is_missing_after_migration:
            return False, "检测到非法操作：安全文件缺失(Error: 0x99)。请联系管理员。"
        else:
            print(">>> [System] 检测到旧版数据，执行 AES 迁移...", flush=True)
            try:
                save_used_hashes([])
                with open(license_path, 'a', encoding='utf-8') as file:
                    file.write(MIGRATION_SIGNATURE)
            except Exception as exc:
                return False, f"初始化失败: {exc}"

    used_list = load_used_hashes()
    if os.path.exists(used_keys_path) and used_list is None:
        return False, "系统数据损坏(Error: 0x93)，请勿修改配置文件。"
    if used_list is None:
        used_list = []
    if not activation_code:
        return False, "激活码不能为空"

    current_hash = hashlib.sha256(activation_code.strip().encode()).hexdigest()
    if current_hash in used_list:
        return False, "该激活码已被使用，无法重复充值。"
    return True, ""


def record_key_usage(activation_code):
    """先持久化充值码哈希，再允许余额增加。"""
    current_hash = hashlib.sha256(activation_code.strip().encode()).hexdigest()
    used_list = load_used_hashes()
    if used_list is None:
        used_list = []
    if current_hash not in used_list:
        used_list.append(current_hash)
        save_used_hashes(used_list)


def modify_string(newline, indices, values):
    newline_list = list(newline)
    values = list(values)
    for index, value in zip(indices, values):
        if 0 <= index < len(newline_list):
            newline_list[index] = value
        else:
            raise ValueError('counter value is too large!')
    return ''.join(newline_list)


def generate_log(file_path, garbage_data: str):
    levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
    messages = [
        "System initialized.", "Connection successful.",
        "File processed successfully.", "Background task completed.",
        "Waiting for user input.", "Update check complete.",
        "Memory usage stable.", "No error detected."
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(garbage_data + "\n")
        for _ in range(50):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} [{random.choice(levels)}] {random.choice(messages)}\n")


def initialize_counter_file():
    counter_file = get_license_file_path()
    if os.path.exists(counter_file):
        return
    os.makedirs(os.path.dirname(counter_file), exist_ok=True)
    line = str(secrets.randbelow(10 ** 32 - 10 ** 31) + 10 ** 31)
    encoded = modify_string(line, COUNTER_INDICES, str(INITIAL_COUNTER).zfill(7))
    generate_log(counter_file, encoded)


def read_counter():
    with open(get_license_file_path(), "r", encoding="utf-8") as file:
        line = file.readlines()[0].strip()
    return int(''.join(line[i] for i in COUNTER_INDICES))


def int_to_base58_length(num: int, length: int = 29) -> str:
    if num < 0:
        raise ValueError("负数无法进行Base58编码")
    chars = []
    while num > 0:
        num, remainder = divmod(num, BASE)
        chars.append(CHARSET[remainder])
    base58_str = ''.join(reversed(chars))
    if len(base58_str) < length:
        base58_str = CHARSET[0] * (length - len(base58_str)) + base58_str
    return base58_str


def encode_all(disk_serial: str, add_count: int, password: str, perm_key=None) -> str:
    _require_private_license_config()
    if perm_key is None:
        perm_key = key29
    random_a = random.getrandbits(8)
    random_b = random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    disk_serial_padded = (disk_serial + '%' * 14)[:14]
    add_count_str = str(add_count).zfill(5)
    data = bytearray([random_a])
    data.extend(random_b.encode('utf-8'))
    data.extend(disk_serial_padded.encode('utf-8'))
    data.extend(add_count_str.encode('utf-8'))
    if len(data) != 21:
        raise ValueError(f"数据长度应为21字节，实际为{len(data)}字节")
    key = hashlib.sha256(password.encode()).digest()[:21]
    encrypted_data = bytearray(data[i] ^ key[i] for i in range(21))
    base58_str = int_to_base58_length(int.from_bytes(encrypted_data, 'big'), 29)
    if len(base58_str) != 29:
        raise ValueError(f"Base58编码后应为29字符，实际为{len(base58_str)}字符")
    permuted_str = ''.join(base58_str[i] for i in perm_key)
    return permuted_str + ''.join(random.choice(CHARSET) for _ in range(3))


def decode_all(code: str, password: str, perm_key=None):
    _require_private_license_config()
    if perm_key is None:
        perm_key = key29
    if len(code) != 32:
        raise ValueError(f"授权码长度必须为32位，实际为{len(code)}位")
    permuted_str = code[:29]
    inv_perm = [0] * 29
    for i, pos in enumerate(perm_key):
        inv_perm[pos] = i
    base58_str = ''.join(permuted_str[i] for i in inv_perm)
    try:
        encrypted_data = fixed_base58_to_int(base58_str).to_bytes(21, 'big')
    except Exception as e:
        raise ValueError(f"Base58解码失败: {e}") from e
    key = hashlib.sha256(password.encode()).digest()[:21]
    data = bytearray(encrypted_data[i] ^ key[i] for i in range(21))
    disk_serial = data[2:16].decode('utf-8', errors='ignore').rstrip('%')
    add_count_str = data[16:21].decode('utf-8', errors='ignore')
    try:
        add_count = int(add_count_str)
    except ValueError as e:
        raise ValueError(f"余量次数格式错误: {add_count_str}") from e
    return disk_serial, add_count


def verify_counter_disk_serial(code_disk_serial, current_disk_serial):
    normalized_current = (str(current_disk_serial) + '%' * 14)[:14].rstrip('%')
    return normalized_current == code_disk_serial


def _show_counter_message():
    _require_private_license_config()
    result = {'count': None}
    root = tk.Tk()
    root.withdraw()
    dialog = tk.Toplevel(root)
    dialog.title(name_tips)
    dialog.geometry("500x300")
    dialog.resizable(False, False)
    disk_serial = _get_disk_serial()
    request_code = encode_all(disk_serial, 0, password3, perm_key=key29)
    tk.Label(dialog, text=name_remain_keys_not_enough).pack(pady=5, padx=20, anchor='w')
    tk.Label(dialog, text=name_counter_license, anchor='w').pack(padx=20, anchor='w')
    code_display = tk.Entry(dialog, width=40, justify='center', font=("Courier", 12))
    code_display.insert(0, request_code)
    code_display.configure(state='readonly')
    code_display.pack(pady=(0, 10))
    tk.Label(dialog, text=name_input_label, anchor='w').pack(pady=(10, 5), padx=20, anchor='w')
    entry = tk.Entry(dialog, width=40, justify='center', font=("Courier", 12))
    entry.pack(pady=(0, 10))

    def on_submit():
        _require_private_license_config()
        code = entry.get().strip()
        is_valid, error_message = check_integrity_and_usability(code)
        if not is_valid:
            messagebox.showerror(name_error, error_message)
            return
        try:
            code_disk_serial, add_count = decode_all(code, password3, perm_key=key29)
            if not verify_counter_disk_serial(code_disk_serial, _get_disk_serial()):
                messagebox.showerror(name_error, "此激活码与当前设备不匹配")
                return
            if not add_count:
                messagebox.showerror(name_error, name_please_input_true_key)
                return
            record_key_usage(code)
            result['count'] = int(add_count)
            messagebox.showinfo(name_tips, f"{name_add_keys_successfully}{add_count}")
            dialog.destroy()
        except Exception as e:
            messagebox.showwarning(name_warning, f"{name_input_error}: {e}")

    tk.Button(dialog, text=name_yes, command=on_submit).pack(pady=10)
    root.wait_window(dialog)
    root.destroy()
    return result['count']


def deal_counter(num):
    combine_file = get_license_file_path()
    try:
        with open(combine_file, "r", encoding="utf-8") as file:
            line = file.readlines()[0].strip()
            data = ''.join(line[i] for i in COUNTER_INDICES)
            data_num = int(data)
    except FileNotFoundError as e:
        messagebox.showerror(name_error, "Security file missing (Error: 0x404).")
        raise Exception("License file deleted during runtime.") from e

    if data_num < num:
        root = tk.Tk()
        root.withdraw()
        while data_num < num:
            try:
                denum = int(_show_counter_message())
                data_num += denum
                newline = str(secrets.randbelow(10 ** 32 - 10 ** 31) + 10 ** 31)
                newstr = modify_string(newline, COUNTER_INDICES, str(data_num).zfill(7))
                generate_log(combine_file, newstr)
            except Exception:
                print("error key!", flush=True)
                continue
        root.destroy()
    minusnum = str(data_num - num).zfill(7)
    minusline = str(secrets.randbelow(10 ** 32 - 10 ** 31) + 10 ** 31)
    minusstr = modify_string(minusline, COUNTER_INDICES, minusnum)
    generate_log(combine_file, minusstr)


# 主验证逻辑
def check_license():
    try:
        if os.path.exists(LICENSE_FILE) and validate_license():
            initialize_counter_file()
            return True
        else:
            auth_code = generate_auth_code()

            while True:
                activation_code = auth_activation_dialog(auth_code)
                if not activation_code:
                    return False

                if verify_activation_code(auth_code, activation_code):
                    create_license_file(auth_code, get_device_hash())
                    initialize_counter_file()
                    root = tk.Tk()
                    root.withdraw()
                    messagebox.showinfo(name_tips, name_license_success)
                    root.destroy()
                    return True
                else:
                    root = tk.Tk()
                    root.withdraw()
                    messagebox.showerror(name_warning, name_license_warning)
                    root.destroy()

    except Exception as e:
        # 不再静默吞掉授权初始化异常，否则程序错误会被误报为“设备未授权”。
        log(f"[LICENSE] 授权校验异常: {type(e).__name__}: {e}")
        return False


# ════════════════════════════════════════════════════════════════
# OCR 辅助函数（从 main_show.py 提取，不依赖 tkinter）
# ════════════════════════════════════════════════════════════════

def _poly_bounds(poly):
    """兼容四点多边形和 [x1, y1, x2, y2] 矩形框。"""
    if poly is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        seq = poly.tolist() if hasattr(poly, "tolist") else list(poly)
    except TypeError:
        return 0.0, 0.0, 0.0, 0.0
    if len(seq) >= 4 and all(isinstance(v, (int, float, np.number)) for v in seq[:4]):
        x1, y1, x2, y2 = [float(v) for v in seq[:4]]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    points = []
    for point in seq:
        try:
            if len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def _as_sequence(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, 'tolist'):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    try:
        return list(value)
    except TypeError:
        return [value]


def _sort_ocr_entries_reading_order(entries):
    """按常见横排文档的阅读顺序排列 OCR 文本框。

    先根据文本框的垂直重叠关系聚合成行，再在行内从左到右排列。
    与单纯按中心点 y 排序相比，这种方式对字号不同、轻微倾斜以及上下标
    更稳定。该函数必须在汇总完所有 OCR 返回批次后再次调用，以保证批次
    之间也遵循同一全局顺序。
    """
    valid = []
    for original_index, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            continue
        try:
            x1 = float(entry.get("x1", 0.0))
            y1 = float(entry.get("y1", 0.0))
            x2 = float(entry.get("x2", x1))
            y2 = float(entry.get("y2", y1))
        except (TypeError, ValueError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        item = dict(entry)
        item.update({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
            "width": x2 - x1, "height": y2 - y1,
            "_reading_order_index": original_index,
        })
        valid.append(item)

    if len(valid) < 2:
        for item in valid:
            item.pop("_reading_order_index", None)
        return valid

    heights = sorted(item["height"] for item in valid if item["height"] > 0)
    median_h = heights[len(heights) // 2] if heights else 0.0
    center_tol = max(6.0, median_h * 0.55)

    # 先粗排后逐个匹配最佳行。不能只与“上一行”比较，否则字号较大的框
    # 容易把相邻两行错误地串在一起。
    valid.sort(key=lambda item: (item["cy"], item["y1"], item["x1"],
                                 item["_reading_order_index"]))
    rows = []
    for item in valid:
        best_row = None
        best_score = None
        for row in rows:
            overlap = max(0.0, min(item["y2"], row["y2"]) - max(item["y1"], row["y1"]))
            min_h = max(1.0, min(item["height"], row["height"]))
            overlap_ratio = overlap / min_h
            center_distance = abs(item["cy"] - row["cy"])
            if overlap_ratio < 0.45 and center_distance > center_tol:
                continue
            score = (overlap_ratio, -center_distance)
            if best_score is None or score > best_score:
                best_row, best_score = row, score
        if best_row is None:
            rows.append({
                "items": [item], "y1": item["y1"], "y2": item["y2"],
                "cy": item["cy"], "height": max(1.0, item["height"]),
            })
            continue
        best_row["items"].append(item)
        # 用中位数更新行中心，避免一个异常高文本框持续拉偏后续判断。
        centers = sorted(member["cy"] for member in best_row["items"])
        best_row["cy"] = centers[len(centers) // 2]
        best_row["y1"] = min(member["y1"] for member in best_row["items"])
        best_row["y2"] = max(member["y2"] for member in best_row["items"])
        row_heights = sorted(max(1.0, member["height"]) for member in best_row["items"])
        best_row["height"] = row_heights[len(row_heights) // 2]

    rows.sort(key=lambda row: (row["cy"], row["y1"]))
    ordered = []
    for row in rows:
        row["items"].sort(key=lambda item: (item["x1"], item["cx"],
                                             item["_reading_order_index"]))
        ordered.extend(row["items"])
    for item in ordered:
        item.pop("_reading_order_index", None)
    return ordered


def _ocr_records_to_markdown_lines(records, excluded_line_ids=None):
    """把有序 OCR 框合并为符合视觉行结构的 Markdown 文本行。"""
    excluded = set(excluded_line_ids or [])
    items = []
    for index, record in enumerate(records or []):
        if not isinstance(record, dict) or record.get("line_id") in excluded:
            continue
        text = str(record.get("text", "")).strip()
        bbox = record.get("bbox")
        if not text or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(bbox[i]) for i in range(4))
        except (TypeError, ValueError):
            continue
        items.append({
            "text": text, "x1": min(x1, x2), "y1": min(y1, y2),
            "x2": max(x1, x2), "y2": max(y1, y2), "index": index,
        })
    if not items:
        return []

    heights = sorted(item["y2"] - item["y1"] for item in items
                     if item["y2"] > item["y1"])
    median_h = heights[len(heights) // 2] if heights else 0.0
    center_tol = max(6.0, median_h * 0.55)
    rows = []
    for item in items:
        item_h = max(1.0, item["y2"] - item["y1"])
        item_cy = (item["y1"] + item["y2"]) / 2.0
        if rows:
            row = rows[-1]
            overlap = max(0.0, min(item["y2"], row["y2"]) - max(item["y1"], row["y1"]))
            overlap_ratio = overlap / max(1.0, min(item_h, row["height"]))
            if overlap_ratio >= 0.45 or abs(item_cy - row["cy"]) <= center_tol:
                row["items"].append(item)
                row["y1"] = min(row["y1"], item["y1"])
                row["y2"] = max(row["y2"], item["y2"])
                centers = sorted((member["y1"] + member["y2"]) / 2.0
                                 for member in row["items"])
                row["cy"] = centers[len(centers) // 2]
                row_heights = sorted(max(1.0, member["y2"] - member["y1"])
                                     for member in row["items"])
                row["height"] = row_heights[len(row_heights) // 2]
                continue
        rows.append({"items": [item], "y1": item["y1"], "y2": item["y2"],
                     "cy": item_cy, "height": item_h})

    markdown_lines = []
    for row in rows:
        row["items"].sort(key=lambda item: (item["x1"], item["index"]))
        markdown_lines.append(" ".join(item["text"] for item in row["items"]))
    return markdown_lines


def _merge_cell_ocr_text(lines):
    """将同一单元格内的完整 OCR 文本按视觉顺序连接，不切割文本。"""
    records = []
    for index, (poly, text) in enumerate(lines or []):
        rect = _poly_bounds(poly)
        records.append({
            "line_id": index, "text": str(text).strip(),
            "bbox": [rect[0], rect[1], rect[2], rect[3]],
        })
    # 同一视觉行保留空格；单元格内的上下换行直接续接。
    return "".join(_ocr_records_to_markdown_lines(records))


def _build_cell_ordered_markdown_lines(raw_lines, cell_records, text_by_num,
                                       assigned_ids, excluded_line_ids=None):
    """先排单元格，再排单元格内部 OCR；未归属文本按坐标保留。"""
    excluded = set(excluded_line_ids or [])
    populated = []
    consumed = set()
    for index, cell in enumerate(cell_records or []):
        text = str(text_by_num.get(cell["number"], "")).strip()
        if not text:
            continue
        rect = cell["rect"]
        populated.append({
            "text": text, "x1": rect[0], "y1": rect[1],
            "x2": rect[2], "y2": rect[3], "index": index,
        })
        consumed.update(assigned_ids.get(cell["number"], []))

    if not populated:
        return _ocr_records_to_markdown_lines(raw_lines, excluded)

    # 单元格按上边界聚合为表格行，再在行内从左到右排列。只比较稳定的
    # 上边界锚点，不通过重叠关系合并或拆分任何单元格。
    heights = sorted(max(1.0, cell["y2"] - cell["y1"]) for cell in populated)
    median_h = heights[len(heights) // 2]
    top_tol = max(6.0, min(18.0, median_h * 0.22))
    populated.sort(key=lambda cell: (cell["y1"], cell["x1"], cell["index"]))
    cell_rows = []
    for cell in populated:
        best = None
        best_distance = None
        for row in cell_rows:
            distance = abs(cell["y1"] - row["top_anchor"])
            if distance <= top_tol and (best_distance is None or distance < best_distance):
                best, best_distance = row, distance
        if best is None:
            cell_rows.append({"cells": [cell], "top_anchor": cell["y1"]})
        else:
            best["cells"].append(cell)
            tops = sorted(item["y1"] for item in best["cells"])
            best["top_anchor"] = tops[len(tops) // 2]

    units = []
    for row in cell_rows:
        row["cells"].sort(key=lambda cell: (cell["x1"], cell["index"]))
        units.append({
            "kind": "cell_row",
            "text": " ".join(cell["text"] for cell in row["cells"]),
            "x1": min(cell["x1"] for cell in row["cells"]),
            "y1": min(cell["y1"] for cell in row["cells"]),
            "x2": max(cell["x2"] for cell in row["cells"]),
            "y2": max(cell["y2"] for cell in row["cells"]),
        })

    # 未归属单元格的 OCR 文本继续保留。先单独恢复普通视觉行，再与表格行
    # 按页面纵坐标混排，避免标题、表外说明和备注丢失。
    remaining = [line for line in (raw_lines or [])
                 if line.get("line_id") not in consumed
                 and line.get("line_id") not in excluded]
    remaining.sort(key=lambda line: (
        float(line.get("bbox", [0, 0, 0, 0])[1]),
        float(line.get("bbox", [0, 0, 0, 0])[0])))
    for line in remaining:
        bbox = line.get("bbox", [0, 0, 0, 0])
        units.append({
            "kind": "raw", "text": str(line.get("text", "")).strip(),
            "x1": float(bbox[0]), "y1": float(bbox[1]),
            "x2": float(bbox[2]), "y2": float(bbox[3]),
        })
    units = [unit for unit in units if unit["text"]]
    units.sort(key=lambda unit: (unit["y1"], unit["x1"]))

    output = []
    pending_raw = []
    def flush_raw():
        if not pending_raw:
            return
        records = [{"line_id": i, "text": unit["text"],
                    "bbox": [unit["x1"], unit["y1"], unit["x2"], unit["y2"]]}
                   for i, unit in enumerate(pending_raw)]
        output.extend(_ocr_records_to_markdown_lines(records))
        pending_raw.clear()

    for unit in units:
        if unit["kind"] == "cell_row":
            flush_raw()
            output.append(unit["text"])
        else:
            pending_raw.append(unit)
    flush_raw()
    return output


def _collect_ordered_entries(result_item):
    rec_texts  = _as_sequence(result_item.get('rec_texts', []))
    rec_scores = _as_sequence(result_item.get('rec_scores', []))
    dt_polys   = _as_sequence(result_item.get('dt_polys', []))
    if not dt_polys:
        dt_polys = _as_sequence(result_item.get('rec_polys', []))
    if not dt_polys:
        dt_polys = _as_sequence(result_item.get('det_polys', []))
    rec_boxes  = _as_sequence(result_item.get('rec_boxes', []))
    if not rec_boxes:
        rec_boxes = _as_sequence(result_item.get('det_boxes', []))
    if len(rec_scores) < len(rec_texts):
        rec_scores = list(rec_scores) + [0.0] * (len(rec_texts) - len(rec_scores))
    entries = []
    for index, text in enumerate(rec_texts):
        text_str = str(text).strip()
        if not text_str:
            continue
        poly = dt_polys[index] if index < len(dt_polys) else (rec_boxes[index] if index < len(rec_boxes) else None)
        x1, y1, x2, y2 = _poly_bounds(poly)
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        entries.append({
            'text': text_str,
            'score': float(rec_scores[index]) if index < len(rec_scores) else 0.0,
            'bbox': poly, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cx': x1 + w / 2.0, 'cy': y1 + h / 2.0, 'width': w, 'height': h,
        })
    return _sort_ocr_entries_reading_order(entries)


def _result_item_to_mapping(item):
    if hasattr(item, 'get'):
        return item
    for attr in ("res", "data"):
        value = getattr(item, attr, None)
        if hasattr(value, 'get'):
            return value
    json_value = getattr(item, "json", None)
    if callable(json_value):
        try:
            json_value = json_value()
        except Exception:
            json_value = None
    if hasattr(json_value, 'get'):
        return json_value
    return None


def _collect_legacy_ocr_entries(raw_result):
    entries = []

    def visit(node):
        if not isinstance(node, (list, tuple)):
            return
        if len(node) >= 2 and isinstance(node[0], (list, tuple)) and isinstance(node[1], (list, tuple)):
            poly = node[0]
            rec = node[1]
            if len(rec) >= 1:
                text = str(rec[0]).strip()
                score = float(rec[1]) if len(rec) > 1 else 0.0
                if text:
                    x1, y1, x2, y2 = _poly_bounds(poly)
                    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                    entries.append({
                        "text": text,
                        "score": score,
                        "bbox": poly,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": x1 + w / 2.0, "cy": y1 + h / 2.0,
                        "width": w, "height": h,
                    })
                    return
        for child in node:
            visit(child)

    visit(raw_result)
    return _sort_ocr_entries_reading_order(entries)


def _extract_ocr_entries(raw_result):
    if not raw_result:
        return []
    entries = []
    items = raw_result if isinstance(raw_result, list) else [raw_result]
    for item in items:
        mapping = _result_item_to_mapping(item)
        if mapping is not None:
            entries.extend(_collect_ordered_entries(mapping))
        elif isinstance(item, (list, tuple)):
            entries.extend(_collect_legacy_ocr_entries(item))
    if not entries:
        entries = _collect_legacy_ocr_entries(raw_result)
    # PaddleOCR/PaddleX 某些版本会按批次返回多个 item；各 item 内部有序并
    # 不代表整页有序，因此这里必须对全部结果再做一次全局阅读顺序排序。
    return _sort_ocr_entries_reading_order(entries)


def _rect_from_position(position):
    if not isinstance(position, (list, tuple)):
        return None
    if len(position) >= 8:
        xs = [float(position[i]) for i in range(0, 8, 2)]
        ys = [float(position[i]) for i in range(1, 8, 2)]
        return min(xs), min(ys), max(xs), max(ys)
    if len(position) == 4:
        x1, y1, x2, y2 = [float(v) for v in position]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return None


def _rect_area(rect):
    if rect is None:
        return 0.0
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _rect_intersection(a, b):
    if a is None or b is None:
        return None
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _point_in_rect(x, y, rect):
    if rect is None:
        return False
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _rect_to_poly(rect):
    x1, y1, x2, y2 = rect
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _canonical_layout_class(class_name):
    name = str(class_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "title": {"title", "doc_title", "paragraph_title", "figure_title"},
        "category_no": {"category_no", "class_no", "classification_no", "archive_no",
                        "catalog_no", "file_no", "record_no", "header"},
        "page_no": {"page_no", "page_number", "pageno", "page", "number"},
        "table": {"table", "table_region", "table_area"},
        "table_cell": {"table_cell", "cell", "tablecell"},
    }
    for canonical, names in aliases.items():
        if name in names:
            return canonical
    return name


def _get_layout_position(box):
    if not isinstance(box, dict):
        return None
    pos = box.get("Position")
    if pos is not None:
        return pos
    coord = box.get("coordinate")
    if isinstance(coord, (list, tuple)) and len(coord) == 4:
        x_min, y_min, x_max, y_max = [float(v) for v in coord]
        return [x_min, y_max, x_max, y_max, x_max, y_min, x_min, y_min]
    return None


def _scale_layout_data(layout_data, scale_x, scale_y):
    """将版式/单元格结果从版式图坐标系转换到 OCR 图坐标系。"""
    if not isinstance(layout_data, dict):
        return layout_data
    try:
        sx, sy = float(scale_x), float(scale_y)
    except (TypeError, ValueError):
        return layout_data
    if sx <= 0 or sy <= 0:
        return layout_data

    def scale_box(box):
        if not isinstance(box, dict):
            return box
        scaled = dict(box)
        pos = box.get("Position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 8:
            scaled["Position"] = [
                float(value) * (sx if index % 2 == 0 else sy)
                for index, value in enumerate(pos)
            ]
        coord = box.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) == 4:
            scaled["coordinate"] = [
                float(coord[0]) * sx, float(coord[1]) * sy,
                float(coord[2]) * sx, float(coord[3]) * sy,
            ]
        cells = box.get("cells")
        if isinstance(cells, list):
            scaled["cells"] = [scale_box(cell) for cell in cells]
        return scaled

    result = dict(layout_data)
    boxes = layout_data.get("Box")
    if isinstance(boxes, list):
        result["Box"] = [scale_box(box) for box in boxes]
    return result


def _expand_layout_boxes(layout_boxes):
    expanded, used = [], set()
    def next_num(prefix="auto"):
        i = 0
        while f"{prefix}_{i}" in used:
            i += 1
        n = f"{prefix}_{i}"; used.add(n); return n
    for bi, box in enumerate(layout_boxes or []):
        if not isinstance(box, dict):
            continue
        base = dict(box)
        num = str(base.get("Number", base.get("number", bi)))
        if num in used:
            num = next_num("box")
        used.add(num); base["Number"] = num
        pos = _get_layout_position(base)
        if pos is not None:
            base["Position"] = pos
        expanded.append(base)
        for ci, cell in enumerate(base.get("cells", []) or []):
            if not isinstance(cell, dict):
                continue
            cb = dict(cell)
            cn = str(cb.get("Number", cb.get("number", f"{num}_c{ci}")))
            if cn in used:
                cn = next_num(f"{num}_cell")
            used.add(cn); cb["Number"] = cn
            cb["Class"] = cb.get("Class", cb.get("class", "table_cell"))
            cb["parent_table"] = cb.get("parent_table", num)
            cp = _get_layout_position(cb)
            if cp is not None:
                cb["Position"] = cp
            expanded.append(cb)
    return expanded


def _build_layout_records(layout_boxes):
    records = []
    parent_tables = {str(b.get("parent_table")) for b in layout_boxes
                     if isinstance(b, dict) and b.get("parent_table") is not None}
    explicit_cell = any(
        _canonical_layout_class(b.get("Class", "")) == "table_cell" or b.get("parent_table") is not None
        for b in layout_boxes if isinstance(b, dict))
    table_like = sum(1 for b in layout_boxes
                     if isinstance(b, dict) and _canonical_layout_class(b.get("Class", "")) == "table")
    for box in layout_boxes:
        if not isinstance(box, dict):
            continue
        pos = _get_layout_position(box)
        rect = _rect_from_position(pos)
        if rect is None or _rect_area(rect) <= 0:
            continue
        num = str(box.get("Number", ""))
        cls = box.get("Class", "")
        canonical = _canonical_layout_class(cls)
        if canonical == "table":
            if box.get("parent_table") is not None:
                canonical = "table_cell"
            elif not explicit_cell and table_like > 1:
                canonical = "table_cell"
        records.append({
            "box": box, "number": num, "class": cls, "canonical_class": canonical,
            "position": pos, "rect": rect,
            "assignable": not (canonical == "table" and num in parent_tables),
        })
    return records


def _ocr_overlap_score(line_rect, target_rect):
    inter = _rect_intersection(line_rect, target_rect)
    if inter is None:
        return 0.0, 0.0, 0.0
    ia = _rect_area(inter)
    return (ia / max(_rect_area(line_rect), 1.0),
            ia / max(_rect_area(target_rect), 1.0), ia)


def _layout_priority(canonical):
    if canonical in ("category_no", "page_no"): return 0
    if canonical == "title": return 1
    if canonical == "table_cell": return 2
    if canonical == "table": return 3
    return 4


def _sort_text_lines(lines, y_threshold=15):
    if not lines:
        return lines
    info = []
    for box, text in lines:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        info.append((min(ys), min(xs), box, text))
    info.sort(key=lambda x: x[0])
    grouped, cur_row, last_y = [], [], None
    for y_min, x_min, box, text in info:
        if last_y is None or abs(y_min - last_y) <= y_threshold:
            cur_row.append((x_min, box, text))
        else:
            cur_row.sort(key=lambda x: x[0])
            grouped.extend(cur_row)
            cur_row = [(x_min, box, text)]
        last_y = y_min
    if cur_row:
        cur_row.sort(key=lambda x: x[0])
        grouped.extend(cur_row)
    return [(b, t) for _, b, t in grouped]


def _fix_horizontal_disorder(lines):
    if not lines or len(lines) < 2:
        return lines
    yc = []; ht = []
    for box, text in lines:
        ys = [p[1] for p in box]
        yc.append((min(ys) + max(ys)) / 2); ht.append(max(ys) - min(ys))
    avg_h = sum(ht) / len(ht) if ht else 0
    avg_y = sum(yc) / len(yc) if yc else 0
    if all(abs(y - avg_y) < avg_h * 1.2 for y in yc):
        return sorted(lines, key=lambda x: min(p[0] for p in x[0]))
    return lines


def _make_ocr_line_record(line_id, entry, text):
    poly = entry.get("bbox")
    rect = _poly_bounds(poly) if poly is not None else (
        float(entry.get("x1", 0)), float(entry.get("y1", 0)),
        float(entry.get("x2", 0)), float(entry.get("y2", 0))
    )
    # rec_boxes/det_boxes 可能是四值矩形；统一输出为四点多边形。
    try:
        seq = poly.tolist() if hasattr(poly, "tolist") else list(poly)
    except (TypeError, AttributeError):
        seq = []
    if len(seq) >= 4 and all(isinstance(v, (int, float, np.number)) for v in seq[:4]):
        poly = _rect_to_poly(rect)
    elif not seq:
        poly = _rect_to_poly(rect)
    else:
        poly = seq
    return {
        "line_id": line_id, "text": text,
        "score": float(entry.get("score", 0.0)) if entry.get("score") is not None else 0.0,
        "poly": [[int(p[0]), int(p[1])] for p in poly],
        "bbox": [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])],
    }


SKIP_OCR_CLASSES = {"figure", "image", "seal", "stamp", "signature", "barcode", "qr_code"}


# ════════════════════════════════════════════════════════════════
# 图像预处理（高清化）
# ════════════════════════════════════════════════════════════════

def _cv2_imread(fp):
    arr = np.fromfile(fp, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _remove_small_noise(mask, min_area=25):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    cleaned = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned

def _split_overlap(image, tile=512):
    """切片并让末块贴边；小于 tile 的图片不再产生负坐标。"""
    h, w = image.shape[:2]
    tile = max(1, int(tile))
    if h <= 0 or w <= 0:
        return [], []

    ys = [0] if h <= tile else list(range(0, h, tile))
    xs = [0] if w <= tile else list(range(0, w, tile))
    patches, coords, seen = [], [], set()
    for y in ys:
        for x in xs:
            l = max(0, min(x, max(0, w - tile)))
            t = max(0, min(y, max(0, h - tile)))
            r = min(w, l + tile)
            b = min(h, t + tile)
            if (l, t) in seen:
                continue
            seen.add((l, t))
            patches.append(image[t:b, l:r])
            coords.append((l, t, r, b))
    return patches, coords

def _merge_prob(patches, coords, hw):
    H, W = hw
    merged = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    for (l, t, r, b), p in zip(coords, patches):
        ph, pw = p.shape
        merged[t:t+ph, l:l+pw] += p
        cnt[t:t+ph, l:l+pw] += 1
    cnt[cnt == 0] = 1
    return merged / cnt

def _adaptive_thresh(prob, bs=51, C=8):
    u8 = (prob * 255).astype(np.uint8)
    return cv2.adaptiveThreshold(u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, bs, C)


# ════════════════════════════════════════════════════════════════
# OCRPipeline — 核心处理流水线
# ════════════════════════════════════════════════════════════════

# imageType 映射：支持数字和字符串
_IMAGE_TYPE_MAP = {
    "1": 1, "raw": 1, "RAW": 1,
    "2": 2, "processed": 2, "PROCESSED": 2,
    "3": 3, "hd": 3, "HD": 3,
}

def _parse_image_type(value) -> int:
    """解析 imageType 参数，支持数字和字符串。"""
    if isinstance(value, int):
        return value if value in (1, 2, 3) else 3
    s = str(value).strip()
    return _IMAGE_TYPE_MAP.get(s, 3)

class _ONNXOCREngine:
    """ONNX OCR 引擎，替代 PaddleOCR 的 det+rec 推理。"""

    def __init__(self, det_onnx_path, rec_onnx_path, session_factory=None):
        import onnxruntime as ort
        factory = session_factory or ort.InferenceSession
        self.det_session = factory(det_onnx_path)
        self.rec_session = factory(rec_onnx_path)

        # 加载字符表并添加 CTCLabelDecode 的特殊字符
        import json, yaml
        rec_dir = os.path.join(os.path.dirname(rec_onnx_path), '..', 'ocr', 'server', 'rec')
        rec_yml_path = os.path.join(rec_dir, 'inference.yml')
        if os.path.isfile(rec_yml_path):
            with open(rec_yml_path, 'r', encoding='utf-8') as f:
                rec_yml = yaml.safe_load(f)
            char_dict = rec_yml['PostProcess']['character_dict']
        else:
            char_dict_path = os.path.join(os.path.dirname(rec_onnx_path), 'char_dict.json')
            with open(char_dict_path, 'r', encoding='utf-8') as f:
                char_dict = json.load(f)
        # PaddleOCR CTCLabelDecode: ['blank'] + character_dict + [' '] → 18383+2=18385 类
        self._char_dict = ['<blank>'] + char_dict + [' ']

    def predict(self, image):
        """执行 OCR 推理，返回与 PaddleOCR 兼容的结果格式。"""
        import onnxruntime as ort
        h, w = image.shape[:2]

        # 1. det 推理 - 文本检测
        det_result = self._det_run(image)
        if not det_result:
            log("[ONNX OCR] det 未检测到文字区域")
            return []

        # 2. rec 推理 - 逐框识别
        # 单个异常/极窄检测框不能拖垮整张图片的 OCR。
        results = []
        skipped = 0
        for box_index, box in enumerate(det_result):
            if len(box) != 4:
                skipped += 1
                continue
            cropped = self._crop_text_region(image, box)
            if cropped is None:
                skipped += 1
                continue
            try:
                text, score = self._rec_run(cropped)
            except Exception as e:
                skipped += 1
                ch, cw = cropped.shape[:2]
                log(
                    f"[WARN] ONNX OCR rec 跳过异常框 index={box_index}, "
                    f"crop={cw}x{ch}, error={type(e).__name__}: {e}"
                )
                continue
            if text:
                # PaddleOCR 兼容格式：[[poly], [text, score]]
                poly = [[int(box[i][0]), int(box[i][1])] for i in range(4)]
                results.append([poly, [text, score]])
        log(
            f"[ONNX OCR] det={len(det_result)}框, "
            f"rec识别={len(results)}条, 跳过={skipped}条"
        )
        return results

    def _det_run(self, image):
        """文本检测推理。返回检测到的文本框列表（四点坐标）。"""
        h, w = image.shape[:2]
        # 预处理：resize 到兼容尺寸（NCHWC 需要 8 的倍数）
        det_limit = 960
        ratio = det_limit / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        # 对齐到 32 的倍数（DB 检测器 stride 要求）
        new_h = max(((new_h + 31) // 32) * 32, 32)
        new_w = max(((new_w + 31) // 32) * 32, 32)

        img = cv2.resize(image, (new_w, new_h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)  # batch

        # 推理
        input_tensor = {self.det_session.get_inputs()[0].name: img}
        output = self.det_session.run(None, input_tensor)[0]

        # DB 后处理（完全对齐 PaddleOCR boxes_from_bitmap 流程）
        # 参数: thresh=0.1, box_thresh=0.3, unclip_ratio=1.2, min_size=3
        import pyclipper

        prob = output[0, 0]  # [H, W]
        pred_h, pred_w = prob.shape[:2]
        # 缩放比例（模型输出尺寸 → 原图尺寸）
        width_scale = w / pred_w
        height_scale = h / pred_h

        # 1. 二值化
        segmentation = prob > 0.1
        mask = segmentation.astype(np.uint8)

        # 2. 查找轮廓（RETR_LIST 与 PaddleOCR 一致）
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            # 3. 获取最小外接矩形（contour 格式为 (N,1,2)，需 reshape 为 (N,2)）
            cnt_2d = cnt.reshape(-1, 2)
            mini_box = cv2.minAreaRect(cnt_2d)
            # 不排序！cv2.boxPoints 返回 [左上, 右上, 右下, 左下]，已经是闭合多边形
            points = list(cv2.boxPoints(mini_box))
            sside = min(mini_box[1])  # 最小边长
            if sside < 3:  # min_size=3
                continue

            points_arr = np.array(points, dtype=np.float32)

            # 4. 计算框内平均得分（box_score_fast）
            xmin = max(0, int(math.floor(points_arr[:, 0].min())))
            xmax = min(pred_w - 1, max(0, int(math.ceil(points_arr[:, 0].max()))))
            ymin = max(0, int(math.floor(points_arr[:, 1].min())))
            ymax = min(pred_h - 1, max(0, int(math.ceil(points_arr[:, 1].max()))))
            if xmax <= xmin or ymax <= ymin:
                continue
            roi = prob[ymin:ymax+1, xmin:xmax+1]
            box_shifted = points_arr.copy()
            box_shifted[:, 0] -= xmin
            box_shifted[:, 1] -= ymin
            fill_mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
            cv2.fillPoly(fill_mask, [box_shifted.astype(np.int32)], 1)
            score = cv2.mean(roi, fill_mask)[0]

            # 5. box_thresh 过滤
            if score < 0.3:
                continue

            # 6. Vatti clipping 膨胀（与 PaddleOCR 完全一致：直接传 float 坐标）
            area = cv2.contourArea(points_arr)
            length = cv2.arcLength(points_arr, True)
            if area < 1 or length < 1:
                continue
            distance = area * 1.2 / length  # unclip_ratio=1.2
            try:
                offset = pyclipper.PyclipperOffset()
                offset.AddPath(points_arr.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                expanded_list = offset.Execute(distance)
                if not expanded_list:
                    continue
                expanded = np.array(expanded_list[0], dtype=np.float32)
            except Exception:
                continue
            if len(expanded) == 0:
                continue

            # 7. 膨胀后再取最小外接矩形
            expanded_box = cv2.minAreaRect(expanded.reshape(-1, 2).astype(np.float32))
            final_points = sorted(list(cv2.boxPoints(expanded_box)), key=lambda x: x[0])
            final_sside = min(expanded_box[1])
            if final_sside < 5:  # min_size + 2
                continue

            # 8. 缩放回原图坐标
            box = np.array(final_points, dtype=np.float32)
            box[:, 0] = np.clip(np.round(box[:, 0] * width_scale), 0, w)
            box[:, 1] = np.clip(np.round(box[:, 1] * height_scale), 0, h)
            boxes.append(box.astype(np.int32))

        # 按阅读顺序排序：先按 y 分组，同组内按 x 排序
        if boxes:
            boxes = list(boxes)
            # 按 y 中心排序
            boxes.sort(key=lambda b: int((b[0, 1] + b[2, 1]) / 2))
            # 合并相近行（y 中心距 < 平均行高一半），每行内按 x 排序
            avg_h = np.mean([b[2, 1] - b[0, 1] for b in boxes])
            merged = []
            row = [boxes[0]]
            for box in boxes[1:]:
                cy0 = int((row[-1][0, 1] + row[-1][2, 1]) / 2)
                cy1 = int((box[0, 1] + box[2, 1]) / 2)
                if abs(cy1 - cy0) < avg_h * 0.8:
                    row.append(box)
                else:
                    row.sort(key=lambda b: b[0, 0])  # 行内按 x 排序
                    merged.extend(row)
                    row = [box]
            row.sort(key=lambda b: b[0, 0])
            merged.extend(row)
            boxes = merged
        return boxes

    def _crop_text_region(self, image, box):
        """裁剪文字区域。"""
        h, w = image.shape[:2]
        x1, y1 = max(0, box[:, 0].min()), max(0, box[:, 1].min())
        x2, y2 = min(w, box[:, 0].max()), min(h, box[:, 1].max())
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _rec_run(self, cropped):
        """文字识别推理。返回识别的文本和置信度。

        ONNX 识别网络会在过窄输入上经过多次下采样，最终产生宽度为 0 的
        中间特征图（典型错误: Invalid input shape: {1,0}）。因此保持原始
        宽高比缩放后，对窄图做右侧白边填充，而不是强行横向拉伸。
        """
        h, w = cropped.shape[:2]
        imgC, imgH = 3, 48
        min_imgW = 32
        max_imgW = 3200  # 与 PaddleOCR max_imgW 一致
        if h <= 0 or w <= 0:
            return "", 0.0

        # 按宽高比计算有效内容宽度。
        ratio = w / float(h)
        content_w = max(1, min(max_imgW, int(math.ceil(imgH * ratio))))

        # Resize；极窄内容保持比例后填充到安全最小宽度，避免网络下采样到 0。
        resized = cv2.resize(cropped, (content_w, imgH))
        if content_w < min_imgW:
            # OpenCV 输入为 BGR，白色背景三通道均为 255。
            padded = np.full((imgH, min_imgW, imgC), 255, dtype=resized.dtype)
            padded[:, :content_w] = resized
            resized = padded

        img = resized.astype(np.float32)
        # 归一化：(x/255 - 0.5) / 0.5
        img = img.transpose((2, 0, 1)) / 255.0
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, 0)

        # ONNX 推理
        input_tensor = {self.rec_session.get_inputs()[0].name: img}
        output = self.rec_session.run(None, input_tensor)[0]

        # CTCLabelDecode: 模型索引直接映射到 self._char_dict
        # _char_dict: ['blank', '　', '一', ..., ' '] — 18385 个
        # 索引 0 = blank（跳过），重复合并
        probs = output[0]
        indices = np.argmax(probs, axis=1)
        scores = np.max(probs, axis=1)

        chars, char_scores = [], []
        prev = -1
        for idx, score in zip(indices, scores):
            if idx == 0:  # blank，跳过，并重置 prev 以允许后续同字符
                prev = -1
                continue
            if idx == prev:  # 重复字符，跳过
                continue
            if 0 < idx < len(self._char_dict):
                chars.append(self._char_dict[idx])
                char_scores.append(score)
            prev = idx

        if not chars:
            return "", 0.0
        return ''.join(chars), float(np.mean(char_scores))


class OCRPipeline:
    """无头版 OCR 处理流水线，复用与 main_show.py 相同的底层模块。"""

    def __init__(self, use_gpu=None, *, onnx_session_factory=None,
                 imgpre_fp16=True, imgpre_batch=None):
        self._onnx_session_factory = onnx_session_factory
        self._imgpre_fp16 = imgpre_fp16
        if imgpre_batch is not None and (type(imgpre_batch) is not int or imgpre_batch < 1):
            raise ValueError('imgpre_batch must be a positive integer')
        self._imgpre_batch = imgpre_batch
        if use_gpu is None:
            try:
                import paddle
                self.use_gpu = paddle.is_compiled_with_cuda() and torch.cuda.is_available()
            except Exception:
                self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu

        self._layout_detector = None
        self._yolo_model = None
        self._table_cell_processor = None
        self._ocr_engine = None
        self._ner_pipeline = None
        self._layout_fusion = None
        self._imgpre_model = None
        self._imgpre_device = None
        self._models_loaded = False
        self._use_onnx_layout = False
        self._use_onnx_cell = False
        self._use_onnx_ocr = False

    # ── 模型加载 ──

    FORCE_PADDLE = os.environ.get("AUTOOCR_FORCE_PADDLE", "") == "1"

    def _try_load_onnx_models(self):
        """尝试加载 ONNX 模型，成功则标记使用 ONNX。"""
        if self.FORCE_PADDLE:
            log("[INFO] AUTOOCR_FORCE_PADDLE=1，强制使用 PaddlePaddle 模型")
            return

        try:
            import onnxruntime as ort
        except ImportError:
            log("[INFO] onnxruntime 未安装，跳过 ONNX 模型")
            return

        onnx_layout = getattr(cfg, 'ONNX_LAYOUT_MODEL', '')
        onnx_cell = getattr(cfg, 'ONNX_CELL_MODEL', '')
        factory = self._onnx_session_factory or ort.InferenceSession

        if onnx_layout and os.path.isfile(onnx_layout):
            try:
                self._layout_detector = factory(onnx_layout)
                self._use_onnx_layout = True
                log(f"[OK] ONNX 版式分析模型已加载: {onnx_layout}")
            except Exception as e:
                log(f"[WARN] ONNX 版式分析模型加载失败: {e}")

        if onnx_cell and os.path.isfile(onnx_cell):
            try:
                self._table_cell_processor = factory(onnx_cell)
                self._use_onnx_cell = True
                log(f"[OK] ONNX 单元格检测模型已加载: {onnx_cell}")
            except Exception as e:
                log(f"[WARN] ONNX 单元格检测模型加载失败: {e}")

        # OCR ONNX 模型
        onnx_ocr_det = getattr(cfg, 'ONNX_OCR_DET_MODEL', '')
        onnx_ocr_rec = getattr(cfg, 'ONNX_OCR_REC_MODEL', '')
        if onnx_ocr_det and os.path.isfile(onnx_ocr_det) and onnx_ocr_rec and os.path.isfile(onnx_ocr_rec):
            try:
                self._ocr_engine = _ONNXOCREngine(onnx_ocr_det, onnx_ocr_rec,
                                               session_factory=self._onnx_session_factory)
                self._use_onnx_ocr = True
                log(f"[OK] ONNX OCR 模型已加载")
            except Exception as e:
                log(f"[WARN] ONNX OCR 模型加载失败: {e}")

    def load_models(self, layout_dir="", cell_dir="", ocr_dir="",
                    ner_dir="", imgpre_dir="./pth/imgpre", yolo_conf=0.3):
        """加载所有模型。路径为空或目录不存在则报错退出。"""
        self.yolo_conf = yolo_conf
        errors = []

        # ── 优先尝试加载 ONNX 模型 ──
        self._try_load_onnx_models()

        # ── 版式分析 (Layout + YOLO) ──
        if self._use_onnx_layout:
            pass  # 已用 ONNX 加载
        elif not layout_dir or not os.path.isdir(layout_dir):
            errors.append(f"版式分析模型目录不存在: {layout_dir}")
        else:
            from layout.paddlepico_detector import PaddlePicoDetector, PaddlePicoDetectorConfig
            det_cfg = PaddlePicoDetectorConfig(
                model_name="PP-DocLayoutV3", model_dir=layout_dir,
                device="gpu" if self.use_gpu else "cpu",
                threshold=0.3, layout_nms=False, layout_unclip_ratio=1.0)
            self._layout_detector = PaddlePicoDetector(det_cfg)
            log("[OK] 版式分析模型已加载")

        # YOLO supplies page/category/photo regions in BOTH layout backends.
        yolo_path = os.path.join(layout_dir, "det_photo", "best1.pt")
        if os.path.isfile(yolo_path):
            from layout.yolo_compat import ensure_yolo_custom_layers
            ensure_yolo_custom_layers()
            from ultralytics import YOLO
            self._yolo_model = YOLO(yolo_path)
        else:
            errors.append(f"YOLO 模型文件不存在: {yolo_path}")

        # ── 单元格检测 ──
        if self._use_onnx_cell:
            pass  # 已用 ONNX 加载
        elif not cell_dir or not os.path.isdir(cell_dir):
            errors.append(f"单元格检测模型目录不存在: {cell_dir}")
        else:
            from layout.table_cell_postprocessor import TableCellPostProcessor, TableCellPostProcessorConfig
            if not self.use_gpu:
                os.environ['FLAGS_use_mkldnn'] = '0'
            cell_cfg = TableCellPostProcessorConfig(
                enabled=True, model_name="RT-DETR-L_wired_table_cell_det",
                model_dir=cell_dir,
                device="gpu" if self.use_gpu else "cpu",
                threshold=0.3, batch_size=1)
            self._table_cell_processor = TableCellPostProcessor(cell_cfg)
            log("[OK] 单元格检测模型已加载")

        # ── OCR 引擎 ──
        if self._use_onnx_ocr:
            pass  # 已用 ONNX 加载
        elif not ocr_dir or not os.path.isdir(ocr_dir):
            errors.append(f"OCR 模型目录不存在: {ocr_dir}")
        else:
            det_d = os.path.join(ocr_dir, "det")
            rec_d = os.path.join(ocr_dir, "rec")
            if not os.path.isdir(det_d) or not os.path.isdir(rec_d):
                errors.append(f"OCR 模型目录结构错误: {ocr_dir}，需要 det/ 和 rec/ 子目录")
            else:
                from paddleocr import PaddleOCR
                base_params = {
                    'use_doc_orientation_classify': False,
                    'use_doc_unwarping': False,
                    'use_textline_orientation': False,
                    'device': "gpu" if self.use_gpu else "cpu",
                    'text_det_thresh': 0.1, 'text_det_box_thresh': 0.3,
                    'text_det_unclip_ratio': 1.2, 'text_det_limit_side_len': 960,
                    'enable_mkldnn': False,
                }
                import yaml
                det_yml = os.path.join(det_d, "inference.yml")
                rec_yml = os.path.join(rec_d, "inference.yml")
                if os.path.isfile(det_yml) and os.path.isfile(rec_yml):
                    with open(det_yml, 'r', encoding='utf-8') as f:
                        det_cfg = yaml.safe_load(f)
                    with open(rec_yml, 'r', encoding='utf-8') as f:
                        rec_cfg = yaml.safe_load(f)
                    det_name = det_cfg.get("Global", {}).get("model_name", "")
                    rec_name = rec_cfg.get("Global", {}).get("model_name", "")
                else:
                    det_name = "PP-OCRv5_server_det"
                    rec_name = "PP-OCRv5_server_rec"
                self._ocr_engine = PaddleOCR(
                    text_detection_model_dir=det_d, text_recognition_model_dir=rec_d,
                    text_detection_model_name=det_name,
                    text_recognition_model_name=rec_name, **base_params)
                log("[OK] OCR 引擎已加载")

        # ── NER ──
        if not ner_dir or not os.path.isdir(ner_dir):
            errors.append(f"NER 模型目录不存在: {ner_dir}")
        else:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline as hf_pipe
            tok = AutoTokenizer.from_pretrained(ner_dir, use_fast=True)
            mdl = AutoModelForTokenClassification.from_pretrained(ner_dir)
            dev = 0 if (self.use_gpu and torch.cuda.is_available()) else -1
            self._ner_pipeline = hf_pipe("token-classification", model=mdl, tokenizer=tok,
                                         aggregation_strategy="simple", device=dev)
            log("[OK] NER 模型已加载")

        # ── 图像预处理（高清化，可选）──
        if imgpre_dir and os.path.isdir(imgpre_dir):
            pth_files = [f for f in os.listdir(imgpre_dir) if f.lower().endswith('.pth')]
            if pth_files:
                try:
                    sys.path.insert(0, imgpre_dir)
                    from train_mobilenetv3_321 import MobileNetV3_Unet
                    model = MobileNetV3_Unet(pretrained=False)
                    self._imgpre_device = torch.device('cuda' if (self.use_gpu and torch.cuda.is_available()) else 'cpu')
                    ckpt = torch.load(os.path.join(imgpre_dir, pth_files[0]),
                                      map_location=self._imgpre_device, weights_only=True)
                    model.load_state_dict(ckpt['G_model'] if 'G_model' in ckpt else ckpt, strict=True)
                    model.to(self._imgpre_device).eval()
                    self._imgpre_model = model
                    log(f"[OK] 图像预处理模型已加载 (设备: {self._imgpre_device})")
                except Exception as e:
                    log(f"[WARN] 图像预处理模型加载失败: {e}")

        # ── 检查是否有致命错误 ──
        if errors:
            log("\n" + "=" * 50)
            log("[FATAL] 以下模型路径错误，服务无法启动：")
            for e in errors:
                log(f"  X {e}")
            log("=" * 50)
            log("请检查 server_config.py 中的模型路径配置，或通过命令行参数指定。")
            log("模型权重不会自动下载，必须手动放置到指定目录。")
            sys.exit(1)

        self._layout_fusion = LayoutFusion(LayoutFusionConfig())
        self._models_loaded = True
        log("[OK] 所有模型加载完成")

    # ── 图像预处理 ──

    def _enhance_single(self, input_path, output_path, tile_size=512, threshold=0.45,
                        local_bs=51, local_C=8, noise_th=15, bg_color=(255, 255, 255), batch=32):
        if self._imgpre_batch is not None:
            batch = min(batch, self._imgpre_batch)
        if self._imgpre_model is None:
            return False
        img = _cv2_imread(input_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches, coords = _split_overlap(rgb, tile_size)
        tf = transforms.Compose([transforms.ToTensor()])
        all_probs = []
        with torch.inference_mode():
            for i in range(0, len(patches), batch):
                bt = torch.stack([tf(p) for p in patches[i:i+batch]]).to(self._imgpre_device)
                if self._imgpre_device.type == 'cuda' and self._imgpre_fp16:
                    bt = bt.half(); self._imgpre_model.half()
                pred = self._imgpre_model(bt)
                all_probs.append(pred[:, 0, :, :].float().cpu().numpy())
                del bt, pred
                if self._imgpre_device.type == 'cuda':
                    torch.cuda.empty_cache()
        prob = _merge_prob(np.concatenate(all_probs, 0), coords, (h, w))
        pc = 1.0 - prob; mask_g = (pc > threshold).astype(np.uint8) * 255; del pc
        mask_l = _adaptive_thresh(prob, local_bs, local_C)
        faint = cv2.subtract(mask_l, mask_g); del mask_l
        faint_c = _remove_small_noise(faint, noise_th); del faint
        final = cv2.bitwise_or(mask_g, faint_c); del mask_g, faint_c
        bmask = cv2.bitwise_not(final); del final
        result = img.copy(); result[bmask == 255] = bg_color
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result[gray > 240] = bg_color; del bmask
        hi, wi = result.shape[:2]; sc = 1500 / max(hi, wi)
        nw, nh = int(wi * sc), int(hi * sc)
        resized = cv2.resize(result, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        del result
        pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)); del resized
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pil.save(output_path, format='JPEG', quality=90, dpi=(300, 300))
        return True

    def _preprocess(self, image_path, work_dir, skip_enhance=False):
        """返回 (resized_path, enhanced_path)

        Args:
            skip_enhance: True 时跳过高清化，仅缩放
        """
        name = os.path.splitext(os.path.basename(image_path))[0]
        r_dir = os.path.join(work_dir, "resized")
        e_dir = os.path.join(work_dir, "enhanced")
        os.makedirs(r_dir, exist_ok=True); os.makedirs(e_dir, exist_ok=True)
        rp = os.path.join(r_dir, f"{name}_resized.jpg")
        ep = os.path.join(e_dir, f"{name}_enhanced.jpg")
        # 缩放每次都执行
        if not os.path.exists(rp):
            with Image.open(image_path) as im:
                # JPEG 输出要求 RGB/L；统一转 RGB，兼容 RGBA、P、CMYK、灰度等输入。
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
                im.save(rp, format='JPEG', quality=90)
        # 高清化根据参数决定
        if not skip_enhance and not os.path.exists(ep):
            self._enhance_single(rp, ep)
        return rp, ep

    # ── 版式分析 ──

    def _detect_cells_onnx(self, detection_path, layout_boxes):
        """ONNX 单元格检测推理。"""
        import onnxruntime as ort
        img = Image.open(detection_path).convert('RGB')
        img_np = np.array(img)

        # 获取表格框
        table_boxes = [b for b in layout_boxes if b.get("label") == "table"]
        if not table_boxes:
            return None

        session = self._table_cell_processor
        input_names = [inp.name for inp in session.get_inputs()]

        tables_result = []
        for tb in table_boxes:
            coord = tb.get("coordinate", [])
            if len(coord) != 4:
                continue
            x1, y1, x2, y2 = [int(c) for c in coord]
            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 预处理
            input_size = 640
            crop_resized = cv2.resize(crop, (input_size, input_size))
            crop_norm = crop_resized.astype(np.float32) / 255.0
            crop_transposed = np.transpose(crop_norm, (2, 0, 1))
            crop_batch = np.expand_dims(crop_transposed, 0)

            # 构建输入
            inputs = {}
            for name in input_names:
                if 'image' in name:
                    inputs[name] = crop_batch.astype(np.float32)
                elif 'im_shape' in name:
                    inputs[name] = np.array([[y2-y1, x2-x1]], dtype=np.float32)
                elif 'scale_factor' in name:
                    inputs[name] = np.array([[(y2-y1)/input_size, (x2-x1)/input_size]], dtype=np.float32)
                else:
                    inputs[name] = np.zeros([1], dtype=np.float32)

            # 推理
            outputs = session.run(None, inputs)

            # 后处理: fetch_name_0=[N,6](class_id,score,x1,y1,x2,y2), fetch_name_1=[1](数量)
            cells = []
            detections = outputs[0]  # [N, 6]
            num_detections = int(outputs[1][0]) if len(outputs) > 1 else len(detections)

            for i in range(min(num_detections, len(detections))):
                row = detections[i]
                class_id = int(row[0])
                score = float(row[1])
                if score < 0.3:
                    continue
                cx1, cy1, cx2, cy2 = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                # 转换回原图坐标
                cx1 = cx1 * (x2-x1) / input_size + x1
                cy1 = cy1 * (y2-y1) / input_size + y1
                cx2 = cx2 * (x2-x1) / input_size + x1
                cy2 = cy2 * (y2-y1) / input_size + y1
                cells.append({
                    "coordinate": [cx1, cy1, cx2, cy2],
                    "score": score
                })

            tables_result.append({
                "table_coordinate": coord,
                "cells": cells,
                "score": float(tb.get("score", 0))
            })

        return {"tables": tables_result} if tables_result else None

    # PP-DocLayoutV3 类别映射（与 inference.yml 中的 label_list 一致）
    _ONNX_LAYOUT_LABELS = {
        0: "abstract", 1: "algorithm", 2: "aside_text", 3: "chart", 4: "content",
        5: "display_formula", 6: "doc_title", 7: "figure_title", 8: "footer",
        9: "footer_image", 10: "footnote", 11: "formula_number", 12: "header",
        13: "header_image", 14: "image", 15: "inline_formula", 16: "number",
        17: "paragraph_title", 18: "reference", 19: "reference_content", 20: "seal",
        21: "table", 22: "text", 23: "vertical_text", 24: "vision_footnote"
    }

    @staticmethod
    def _nms_boxes(boxes, iou_threshold=0.5):
        """对检测框按类别做 NMS（非极大值抑制）。"""
        if not boxes:
            return []

        # 按 score 降序排序
        sorted_boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)
        kept = []

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a["coordinate"]
            bx1, by1, bx2, by2 = b["coordinate"]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter / (area_a + area_b - inter + 1e-6)

        while sorted_boxes:
            best = sorted_boxes.pop(0)
            kept.append(best)
            sorted_boxes = [
                b for b in sorted_boxes
                if not (b["label"] == best["label"] and iou(best, b) > iou_threshold)
            ]

        return kept

    def _detect_layout_onnx(self, detection_path, original_path):
        """ONNX 版式分析推理。"""
        try:
            import onnxruntime as ort
            img = Image.open(detection_path).convert('RGB')
            img_np = np.array(img)
            h, w = img_np.shape[:2]

            # 预处理：resize 到 800x800，归一化
            input_size = 800
            img_resized = cv2.resize(img_np, (input_size, input_size))
            img_norm = img_resized.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
            img_batch = np.expand_dims(img_transposed, 0)  # 添加 batch 维度

            # 获取输入名
            session = self._layout_detector
            input_names = [inp.name for inp in session.get_inputs()]

            # 构建输入
            inputs = {}
            for name in input_names:
                if 'image' in name:
                    inputs[name] = img_batch.astype(np.float32)
                elif 'im_shape' in name:
                    inputs[name] = np.array([[h, w]], dtype=np.float32)
                elif 'scale_factor' in name:
                    inputs[name] = np.array([[h / input_size, w / input_size]], dtype=np.float32)

            # 推理
            outputs = session.run(None, inputs)

            # 后处理：解析检测结果
            # 输出格式: fetch_name_0=[N,7](class_id,score,x1,y1,x2,y2,?), fetch_name_1=[1](数量)
            raw_boxes = []
            detections = outputs[0]  # [N, 7]
            num_detections = int(outputs[1][0]) if len(outputs) > 1 else len(detections)
            conf_threshold = 0.3  # 置信度阈值

            for i in range(min(num_detections, len(detections))):
                row = detections[i]
                class_id = int(row[0])
                score = float(row[1])
                if score < conf_threshold:
                    continue  # 过滤低置信度
                x1, y1, x2, y2 = float(row[2]), float(row[3]), float(row[4]), float(row[5])

                # 映射类别名
                label = self._ONNX_LAYOUT_LABELS.get(class_id, f"class_{class_id}")

                # 缩放回原图坐标
                x1 = x1 * w / input_size
                y1 = y1 * h / input_size
                x2 = x2 * w / input_size
                y2 = y2 * h / input_size

                raw_boxes.append({
                    "label": label,
                    "score": score,
                    "coordinate": [x1, y1, x2, y2]
                })

            # NMS 去重
            boxes = self._nms_boxes(raw_boxes, iou_threshold=0.5)
            log(f"[ONNX] 原始检测={len(raw_boxes)}个, NMS后={len(boxes)}个")

            # 使用 layout_fusion 映射到标准类别
            yolo_res = None
            if self._yolo_model:
                try:
                    dev = 'cuda:0' if (self.use_gpu and torch.cuda.is_available()) else 'cpu'
                    yolo_res = self._yolo_model.predict(str(detection_path), conf=self.yolo_conf,
                                                        verbose=False, device=dev)
                except Exception:
                    pass

            cell_payload = None
            if self._use_onnx_cell and self._table_cell_processor:
                try:
                    cell_payload = self._detect_cells_onnx(detection_path, boxes)
                except Exception:
                    pass

            img_size = None
            try:
                with Image.open(detection_path) as im:
                    img_size = im.size
            except Exception:
                pass

            std_boxes = self._layout_fusion.fuse(
                layout_boxes=boxes, yolo_results=yolo_res,
                table_cell_payload=cell_payload, image_size=img_size)
            return {"Imagepage": os.path.basename(original_path), "Box": std_boxes}

        except Exception as e:
            log(f"[ERROR] ONNX 版式分析失败: {e}")
            return {"Imagepage": os.path.basename(original_path), "Box": []}

    def _detect_layout(self, detection_path, original_path):
        if self._layout_detector is None:
            return None

        # ONNX 推理
        if self._use_onnx_layout:
            return self._detect_layout_onnx(detection_path, original_path)

        # Paddle 推理
        raw = self._layout_detector.predict(Path(detection_path), batch_size=1)
        if not raw:
            return {"Imagepage": os.path.basename(original_path), "Box": []}
        rd = self._layout_detector.result_to_dict(raw[0])
        layout_boxes = self._layout_detector.extract_boxes(rd)

        cell_payload = None
        if self._use_onnx_cell and self._table_cell_processor:
            # ONNX 单元格检测
            try:
                cell_payload = self._detect_cells_onnx(detection_path, layout_boxes)
            except Exception as e:
                log(f"[ERROR] ONNX 单元格检测失败: {e}")
        elif self._table_cell_processor:
            tbs = self._table_cell_processor.pick_table_boxes(layout_boxes)
            if tbs:
                cell_payload = self._table_cell_processor.detect_cells(
                    image_path=Path(detection_path), table_boxes=tbs)

        yolo_res = None
        if self._yolo_model:
            try:
                dev = 'cuda:0' if (self.use_gpu and torch.cuda.is_available()) else 'cpu'
                yolo_res = self._yolo_model.predict(str(detection_path), conf=self.yolo_conf,
                                                    verbose=False, device=dev)
            except Exception:
                pass

        img_size = None
        try:
            with Image.open(detection_path) as im:
                img_size = im.size
        except Exception:
            pass

        std_boxes = self._layout_fusion.fuse(
            layout_boxes=layout_boxes, yolo_results=yolo_res,
            table_cell_payload=cell_payload, image_size=img_size)
        return {"Imagepage": os.path.basename(original_path), "Box": std_boxes}

    # ── OCR 识别 ──

    def _run_ocr(self, image_path, resized_path, layout_data):
        img = Image.open(resized_path).convert('RGB')
        log(f"[_run_ocr] 图片尺寸: {img.size}")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        t0 = time.time()
        log(f"[_run_ocr] 开始OCR推理...")
        result = self._ocr_engine.predict(bgr)
        log(f"[_run_ocr] OCR推理完成")
        entries = _extract_ocr_entries(result)
        if not entries:
            try:
                result = self._ocr_engine.predict(str(resized_path))
                entries = _extract_ocr_entries(result)
            except Exception:
                pass
        if not entries and hasattr(self._ocr_engine, "ocr"):
            try:
                result = self._ocr_engine.ocr(str(resized_path), cls=False)
                entries = _extract_ocr_entries(result)
            except Exception:
                pass
        elapsed = int((time.time() - t0) * 1000)

        raw_lines = []
        for e in entries:
            # 保留识别文本内部空格，只清理首尾空白，避免破坏英文/编号内容。
            t = str(e.get('text', '')).strip()
            if t and float(e.get('score', 0)) > 0.03:
                raw_lines.append(_make_ocr_line_record(len(raw_lines), e, t))

        img_name = os.path.basename(image_path)
        boxes = layout_data.get("Box", []) if layout_data else []
        expanded = _expand_layout_boxes(boxes)
        records = _build_layout_records(expanded)
        box_texts = {r["number"]: [] for r in records if r["assignable"]}
        assigned_ids = {n: [] for n in box_texts}
        special = {"title": [], "category_no": [], "page_no": []}
        assignable = [r for r in records if r["assignable"] and r["class"] not in SKIP_OCR_CLASSES]
        cells = [r for r in assignable if r["canonical_class"] == "table_cell"]
        non_cells = [r for r in assignable if r["canonical_class"] != "table_cell"]

        for line in raw_lines:
            lr = _rect_from_position(line.get("bbox"))
            if lr is None or _rect_area(lr) <= 0:
                continue
            cx = (lr[0] + lr[2]) / 2.0; cy = (lr[1] + lr[3]) / 2.0
            # OCR 文本保持完整且只归属一个单元格。中心点包含优先；若中心点
            # 因检测误差刚好越界，仅在单元格覆盖 OCR 框过半时才兜底归属。
            cell_candidates = []
            for c in cells:
                lo, bo, ia = _ocr_overlap_score(lr, c["rect"])
                center_inside = _point_in_rect(cx, cy, c["rect"])
                if center_inside or lo >= 0.50:
                    cell_candidates.append({
                        "r": c, "inside": center_inside, "lo": lo,
                        "ia": ia, "area": _rect_area(c["rect"]),
                    })
            if cell_candidates:
                cell_candidates.sort(key=lambda item: (
                    -int(item["inside"]), -item["lo"], item["area"],
                    item["r"]["rect"][1], item["r"]["rect"][0]))
                r = cell_candidates[0]["r"]
                n = r["number"]
                box_texts.setdefault(n, []).append((line.get("poly", []), line["text"]))
                assigned_ids.setdefault(n, []).append(line["line_id"])
                continue

            # 未落入单元格的文本再匹配标题、分类号、页码及普通版式框。
            cands = []
            for r in non_cells:
                lo, bo, ia = _ocr_overlap_score(lr, r["rect"])
                cp = _canonical_layout_class(r["class"])
                thr = 0.35 if cp in ("title", "category_no", "page_no") else (0.20 if cp in ("table", "table_cell") else 0.30)
                if lo >= thr or (_point_in_rect(cx, cy, r["rect"]) and lo > 0):
                    cands.append({"r": r, "lo": lo, "bo": bo, "ia": ia, "p": _layout_priority(cp)})
            if not cands:
                continue
            cands.sort(key=lambda x: (x["p"], -x["lo"], -x["bo"], -x["ia"]))
            best = cands[0]; r = best["r"]; n = r["number"]
            box_texts.setdefault(n, []).append((line.get("poly", []), line["text"]))
            assigned_ids.setdefault(n, []).append(line["line_id"])
            cn = r["canonical_class"]
            if cn in special:
                special[cn].append({
                    "text": line["text"],
                    "bbox": line["bbox"],
                    "line_id": line.get("line_id"),
                    "layout_number": n,
                    "layout_class": r["class"],
                })

        text_by_num = {}
        for num, lines in box_texts.items():
            if lines:
                text_by_num[num] = _merge_cell_ocr_text(lines)
            else:
                text_by_num[num] = ""

        boxes_out, text_groups, fp, co = [], [], [], 0
        for box in expanded:
            num = str(box["Number"]); cls = box.get("Class", "")
            nb = dict(box); txt = text_by_num.get(num, "")
            if cls in SKIP_OCR_CLASSES:
                nb["char_start"] = -1; nb["char_end"] = -1
            else:
                nb["char_start"] = co; nb["char_end"] = co + len(txt)
                fp.append(txt); co += len(txt) + 1
            if "Position" in nb:
                nb["Position"] = [int(v) for v in nb["Position"]]
            boxes_out.append(nb)
            text_groups.append({"Content": [txt], "Number": [num], "Class": cls})

        assigned_line_ids = {line_id for ids in assigned_ids.values() for line_id in ids}
        assigned_parts = [txt for txt in fp if txt]
        unassigned_parts = [
            line["text"] for line in raw_lines
            if line.get("line_id") not in assigned_line_ids and line.get("text")
        ]
        raw_full_text = " ".join(line["text"] for line in raw_lines if line.get("text"))
        full_text = " ".join(assigned_parts + unassigned_parts) if assigned_parts else raw_full_text
        markdown_body_lines = _build_cell_ordered_markdown_lines(
            raw_lines, cells, text_by_num, assigned_ids,
            _layout_special_line_ids(special),
        )
        if raw_full_text and not any(group.get("Content", [""])[0] for group in text_groups):
            text_groups.append({"Content": [raw_full_text], "Number": ["raw_ocr"], "Class": "raw_ocr"})
        return {
            "Imagepage": img_name, "Box": boxes_out, "Text": text_groups,
            "FullText": full_text, "elapsed_ms": elapsed,
            "overall_confidence": float(sum(float(e.get('score', 0)) for e in entries) / len(entries)) if entries else 0.0,
            "items": [{"text": e.get('text', ''), "score": float(e.get('score', 0))} for e in entries],
            "RawLines": raw_lines,
            "MarkdownBodyLines": markdown_body_lines,
            "special": special,
        }

    # ── NER 提取 ──

    def _run_ner(self, full_text):
        if self._ner_pipeline is None or not full_text:
            return {}
        return predict_fields_ner(self._ner_pipeline, full_text, list(VALID_LABELS),
                                  score_threshold=cfg.NER_SCORE_THRESHOLD)

    # ── 主入口 ──

    def process(self, image_path: str, work_dir: str = None, image_type=3, run_ner=True) -> dict:
        """处理单张图片，返回完整结果字典。

        Args:
            image_path: 图片路径
            work_dir: 工作目录（None 则自动生成并处理完后删除）
            image_type: 图片类型（支持数字或字符串）
                1 / RAW       = 缩放 + 高清化
                2 / PROCESSED = 缩放 + 高清化
                3 / HD        = 仅缩放，跳过高清化（默认）
            run_ner: 是否在本次流水线中执行 NER。/OCR 应设为 False，
                避免 OCR 接口做字段提取工作。
        """
        image_type = _parse_image_type(image_type)
        if not self._models_loaded:
            raise RuntimeError("模型尚未加载，请先调用 load_models()")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")

        _auto_clean = False
        if work_dir is None:
            work_dir = os.path.join(cfg.TEMP_DIR, uuid.uuid4().hex[:8])
            _auto_clean = True
        os.makedirs(work_dir, exist_ok=True)

        try:
            t_start = time.time()

            # 1. 先只做等比缩放。Layout 和 Cell 必须在高清化之前运行，避免
            # 背景白化、线条增强等操作改变版式模型看到的原始结构。
            t1 = time.time()
            resized_path, enhanced_path = self._preprocess(
                image_path, work_dir, skip_enhance=True)
            with Image.open(resized_path) as _layout_im:
                layout_w, layout_h = _layout_im.size
            t_resize = time.time() - t1

            # 2. 在未高清化的等比缩放图上执行版式分析和单元格检测。
            t2 = time.time()
            layout_data = self._detect_layout(resized_path, image_path)
            t_layout = round(time.time() - t2, 2)
            boxes = layout_data.get('Box', [])
            # 统计各类别数量
            table_count = sum(1 for b in boxes if b.get('Class') == 'table')
            cell_count = sum(1 for b in boxes if b.get('Class') == 'table_cell')
            other_count = len(boxes) - table_count - cell_count
            log(f"[process] 版式分析完成, 耗时={t_layout}s")
            log(f"[process]   回归框={other_count}个 (layout+yolo, 不含table)")
            log(f"[process]   table={table_count}个, cell={cell_count}个")

            # 3. 版式分析完成后再高清化，OCR 优先使用高清化图。高清化输出
            # 可能与版式图尺寸不同，因此显式记录两个坐标系的缩放比例。
            t3 = time.time()
            if image_type == 3:
                log("[process] image_type=3, 版式分析后跳过高清化")
                detection_path = resized_path
            else:
                log(f"[process] image_type={image_type}, 版式分析后执行高清化")
                if not os.path.isfile(enhanced_path):
                    self._enhance_single(resized_path, enhanced_path)
                detection_path = enhanced_path if os.path.isfile(enhanced_path) else resized_path
            t_enhance = time.time() - t3
            t_preprocess = round(t_resize + t_enhance, 2)

            with Image.open(detection_path) as _processed_im:
                processed_w, processed_h = _processed_im.size
            layout_to_ocr_x = processed_w / layout_w if layout_w > 0 else 1.0
            layout_to_ocr_y = processed_h / layout_h if layout_h > 0 else 1.0
            if abs(layout_to_ocr_x - 1.0) > 1e-6 or abs(layout_to_ocr_y - 1.0) > 1e-6:
                layout_data = _scale_layout_data(
                    layout_data, layout_to_ocr_x, layout_to_ocr_y)
                log(
                    f"[process] 版式坐标缩放到OCR图: "
                    f"{layout_w}x{layout_h} -> {processed_w}x{processed_h}, "
                    f"scale=({layout_to_ocr_x:.6f}, {layout_to_ocr_y:.6f})"
                )

            # 4. OCR
            t3 = time.time()
            ocr_result = self._run_ocr(image_path, detection_path, layout_data)
            t_ocr = round(time.time() - t3, 2)
            log(f"[process] OCR完成, 行数={len(ocr_result.get('RawLines', []))}, 耗时={t_ocr}s")

            # 5. NER（OCR 接口按协议只做 OCR，因此可跳过）
            t4 = time.time()
            ner_fields = self._run_ner(ocr_result.get("FullText", "")) if run_ner else {}
            t_ner = round(time.time() - t4, 2)
            if run_ner:
                log(f"[process] NER完成, 字段={len(ner_fields)}个, 耗时={t_ner}s")
            else:
                log("[process] 按接口职责跳过NER")

            total_sec = round(time.time() - t_start, 2)
            log(f"[process] 总耗时={total_sec}s (预处理={t_preprocess}s, 版式={t_layout}s, OCR={t_ocr}s, NER={t_ner}s)")

            return {
                "success": True,
                "data": {
                    "Imagepage": ocr_result["Imagepage"],
                    "Box": ocr_result["Box"],
                    "Text": ocr_result["Text"],
                    "FullText": ocr_result["FullText"],
                    "RawLines": ocr_result["RawLines"],
                    "MarkdownBodyLines": ocr_result.get("MarkdownBodyLines", []),
                    "NER": ner_fields,
                    "special": ocr_result.get("special", {}),
                    # 不能返回临时文件路径：work_dir 可能在 return 前的 finally 中删除。
                    "_processed_width": processed_w,
                    "_processed_height": processed_h,
                },
                "elapsed_sec": total_sec,
            }
        finally:
            if _auto_clean:
                shutil.rmtree(work_dir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════
# 辅助函数：生成有序 JSON 响应
# ════════════════════════════════════════════════════════════════

def _json_response(data, status_code=200):
    """返回保持字段顺序的 JSON 响应"""
    return Response(
        json.dumps(data, ensure_ascii=False),
        status=status_code,
        content_type='application/json; charset=utf-8'
    )


# ════════════════════════════════════════════════════════════════
# Flask 应用
# ════════════════════════════════════════════════════════════════

# 联机版应用（端口5051）
app_online = Flask(__name__)

pipeline: OCRPipeline = None  # type: ignore



def _error_response(status: str, message: str):
    """协议统一错误响应；HTTP 状态码始终保持 200。"""
    return _json_response({"isSuccess": False, "status": str(status), "msg": {"error": message}})


def _valid_image_type(value) -> bool:
    """V1.1 对外协议使用字符串枚举值 "1" / "2" / "3"。"""
    return isinstance(value, str) and value in ("1", "2", "3")


def _normalize_ocr_payload(ocr_result):
    """兼容传入 OCR msg 或完整 OCR 响应，返回标准 msg 对象。"""
    if not isinstance(ocr_result, dict):
        raise ValueError("ocrResult 必须是对象")
    if "isSuccess" in ocr_result and ocr_result.get("isSuccess") is False:
        raise ValueError("ocrResult 表示 OCR 识别失败")
    payload = ocr_result.get("msg") if "msg" in ocr_result else ocr_result
    if not isinstance(payload, dict):
        raise ValueError("ocrResult.msg 必须是对象")
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("ocrResult 缺少非空 pages 数组")
    for i, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"ocrResult.pages[{i}] 必须是对象")
    return payload


def _field_aliases(*values):
    """从字段名/描述生成用于动态标签抽取的候选别名。"""
    aliases = []
    removable = (
        "档案中", "档案", "人员", "材料", "当前页", "右下角", "手写",
        "字段", "中文", "内容", "名称", "编号值", "的",
    )
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        candidates = [text]
        simplified = text
        for token in removable:
            simplified = simplified.replace(token, "")
        simplified = simplified.strip(" ：:，,。.;；()（）[]【】")
        if simplified:
            candidates.append(simplified)
        for candidate in candidates:
            if candidate and candidate not in aliases:
                aliases.append(candidate)
    return aliases


def _extract_labeled_value(text: str, aliases, all_aliases=None) -> str:
    """从 OCR 文本中按“字段名: 值”动态提取，支持多个字段位于同一行。"""
    if not text:
        return ""
    targets = [a for a in aliases if a]
    if not targets:
        return ""
    all_labels = [a for a in (all_aliases or targets) if a]
    # 长标签优先，避免“姓名”抢先匹配“人员姓名”等更具体标签。
    all_labels = sorted(set(all_labels), key=len, reverse=True)
    next_label = "|".join(re.escape(a) for a in all_labels)
    for alias in sorted(set(targets), key=len, reverse=True):
        # 首选明确分隔符；终点是下一个已知“标签:”或换行/文本结束。
        if next_label:
            pattern = rf"{re.escape(alias)}\s*[:：]\s*(.+?)(?=(?:{next_label})\s*[:：]|\n|$)"
        else:
            pattern = rf"{re.escape(alias)}\s*[:：]\s*(.+?)(?=\n|$)"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" \t:：,，;；")

        # OCR 常把冒号漏掉；仅对“行首标签+值”做保守兜底。
        pattern_no_colon = rf"(?:^|\n)\s*{re.escape(alias)}\s+([^\n]{{1,120}})(?=\n|$)"
        m = re.search(pattern_no_colon, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _match_ner_value(ner_result, aliases) -> str:
    """非破坏性匹配 NER 结果，避免顶层字段消费掉 NER 字段。"""
    if not isinstance(ner_result, dict):
        return ""
    for alias in aliases:
        value = ner_result.get(alias)
        if value not in (None, ""):
            return str(value).strip()
    for alias in aliases:
        for key, value in ner_result.items():
            key_s = str(key)
            if alias and (alias in key_s or key_s in alias) and value not in (None, ""):
                return str(value).strip()
    return ""


def _semantic_kind(name: str, description: str) -> str:
    probe = f"{name} {description}".lower()
    if "页码" in probe or "pagenumber" in probe or "page number" in probe or name.lower() in ("page", "pageno"):
        return "pagenumber"
    if "分类" in probe or name.lower() in ("class", "category", "categoryno", "classno"):
        return "class"
    if "标题" in probe or name.lower() in ("title", "documenttitle", "doctitle"):
        return "title"
    return ""


def _normalize_pagenumber(value) -> str:
    """页码只保留 ASCII 数字，无法识别时返回空字符串。"""
    return re.sub(r"[^0-9]", "", str(value or ""))


def _read_markdown_header(markdown: str):
    """读取 Markdown 固定前三行元数据，并返回剩余正文。

    行 1: title
    行 2: class
    行 3: pagenumber

    缺失或空白行统一返回空字符串，不再通过位置、正则或正文内容推断。
    """
    lines = (markdown or "").splitlines()
    if len(lines) < 3:
        lines.extend([""] * (3 - len(lines)))
    metadata = {
        "title": lines[0].strip(),
        "class": lines[1].strip(),
        "pagenumber": lines[2].strip(),
    }
    body = "\n".join(lines[3:])
    return metadata, body


def _layout_special_text(special, key: str) -> str:
    """按版式框阅读顺序合并某类特殊区域文本，并压成单行。"""
    if not isinstance(special, dict):
        return ""
    items = special.get(key, [])
    if not isinstance(items, list):
        return ""

    ordered = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        value = str(item.get("text", "")).replace("\r", " ").replace("\n", " ").strip()
        if not value:
            continue
        bbox = item.get("bbox", [0, 0, 0, 0])
        try:
            x1 = float(bbox[0]) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else 0.0
            y1 = float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else 0.0
        except (TypeError, ValueError):
            x1, y1 = 0.0, 0.0
        ordered.append((y1, x1, index, value))

    ordered.sort(key=lambda row: (row[0], row[1], row[2]))
    return "".join(row[3] for row in ordered)


def _layout_special_line_ids(special):
    """返回已作为 Markdown 头部输出的版式特殊文本行 ID。"""
    if not isinstance(special, dict):
        return set()
    line_ids = set()
    for key in ("title", "category_no", "page_no"):
        items = special.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get("line_id") is not None:
                line_ids.add(item.get("line_id"))
    return line_ids


def _convert_top_value(value, field_type):
    """按协议类型转换；转换失败返回 None，绝不伪造 0/False。"""
    if value in (None, ""):
        return ""
    if field_type == "String":
        return str(value)
    if field_type == "Number":
        text = str(value).strip().replace(",", "")
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", text):
            return float(text)
        return None
    if field_type == "Boolean":
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "是", "真"):
            return True
        if text in ("false", "0", "no", "否", "假"):
            return False
        return None
    return None


@app_online.route('/OCR', methods=['POST'])
def api_ocr():
    """OCR 识别接口：接收 Base64 单图，返回文本与原图像素坐标。"""
    t0 = time.monotonic()
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return _error_response("400", "请求体必须是 JSON 对象")

    if "img" not in data:
        return _error_response("400", "缺少 img 参数")
    if "imageType" not in data:
        return _error_response("400", "缺少 imageType 参数")

    b64 = data.get("img")
    image_type_raw = data.get("imageType")
    if not isinstance(b64, str):
        return _error_response("400", "img 必须是 Base64 字符串")
    if not _valid_image_type(image_type_raw):
        return _error_response("400", f"imageType 值无效: {image_type_raw}，应为字符串 1、2 或 3")
    if not b64:
        return _error_response("400", "img 不能为空")

    # V1.1 明确禁止 data:image/...;base64, 前缀和任何空白字符。
    if b64.lower().startswith("data:"):
        return _error_response("400", "Base64 编码禁止携带 data:image/...;base64, 前缀")
    if any(ch.isspace() for ch in b64):
        return _error_response("400", "Base64 编码不能包含空格、换行符或其他空白字符")
    if len(b64) < 16:
        return _error_response("400", "Base64 数据过短，非有效图片")

    image_type = _parse_image_type(image_type_raw)
    log(f"[OCR] 请求: base64长度={len(b64)} imageType={image_type}")

    try:
        img_bytes = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return _error_response("400", "img 不是合法的 Base64 编码")
    if not img_bytes:
        return _error_response("400", "Base64 解码结果为空")

    tmp_dir = os.path.join(cfg.TEMP_DIR, "upload")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.img")
    try:
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)

        # 先 verify 再重新打开取尺寸；无法解析的图片按协议返回 404。
        try:
            with Image.open(tmp_path) as im:
                im.verify()
            with Image.open(tmp_path) as im:
                orig_w, orig_h = im.size
        except (UnidentifiedImageError, OSError, ValueError) as e:
            log(f"[OCR] 图片解析失败: {e}")
            return _error_response("404", "图片无法解析")

        if pipeline is None:
            return _error_response("500", "OCR 服务尚未完成模型初始化")

        # OCR 与字段提取解耦：本接口不执行 NER。
        result = pipeline.process(tmp_path, image_type=image_type, run_ner=False)
        elapsed = time.monotonic() - t0
        if elapsed > 180:
            log(f"[OCR] 超时: 耗时={elapsed:.2f}s > 180s")
            return _error_response("408", "OCR识别超时")
        if not isinstance(result, dict) or not result.get("success"):
            return _error_response("404", "OCR识别失败")

        result_data = result.get("data", {})
        processed_w = result_data.get("_processed_width") or orig_w
        processed_h = result_data.get("_processed_height") or orig_h
        try:
            processed_w = float(processed_w)
            processed_h = float(processed_h)
        except (TypeError, ValueError):
            processed_w, processed_h = float(orig_w), float(orig_h)
        scale_x = orig_w / processed_w if processed_w > 0 else 1.0
        scale_y = orig_h / processed_h if processed_h > 0 else 1.0

        raw_lines = result_data.get("RawLines", [])
        if not isinstance(raw_lines, list):
            raw_lines = []

        # Markdown 固定前三行由版式分析结果提供：title、class、pagenumber。
        # 任一版式类别未识别到文本时，对应行保留为空字符串。
        special = result_data.get("special", {})
        layout_title = _layout_special_text(special, "title")
        layout_class = _layout_special_text(special, "category_no")
        layout_page_number = _layout_special_text(special, "page_no")
        header_line_ids = _layout_special_line_ids(special)

        blocks = []
        markdown_lines = [layout_title, layout_class, layout_page_number]
        for line in raw_lines:
            if not isinstance(line, dict):
                continue
            text = str(line.get("text", ""))
            bbox = line.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                x1 = int(round(float(bbox[0]) * scale_x))
                y1 = int(round(float(bbox[1]) * scale_y))
                x2 = int(round(float(bbox[2]) * scale_x))
                y2 = int(round(float(bbox[3]) * scale_y))
            except (TypeError, ValueError):
                continue
            # 约束到原图边界，防止缩放/检测浮点误差产生越界坐标。
            x1, x2 = sorted((max(0, min(orig_w, x1)), max(0, min(orig_w, x2))))
            y1, y2 = sorted((max(0, min(orig_h, y1)), max(0, min(orig_h, y2))))
            blocks.append({"text": text, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        # 版式特殊字段已经放入前三行；正文按物理文本行合并，避免同一行的
        # “字段名”和“字段值”被拆成多行，从而为 NER 保留更自然的上下文。
        structured_body = result_data.get("MarkdownBodyLines")
        if isinstance(structured_body, list):
            markdown_lines.extend(str(line) for line in structured_body if str(line).strip())
        else:
            markdown_lines.extend(_ocr_records_to_markdown_lines(raw_lines, header_line_ids))

        msg = {
            "total_pages": 1,
            "pages": [{
                "page_num": 1,
                "width": orig_w,
                "height": orig_h,
                "markdown": "\n".join(markdown_lines),
                "blocks": blocks,
            }],
        }
        elapsed = time.monotonic() - t0
        log(f"[OCR] 成功: 耗时={elapsed:.2f}s, blocks={len(blocks)}条")
        return _json_response({"isSuccess": True, "status": "200", "msg": msg})
    except TimeoutError:
        return _error_response("408", "OCR识别超时")
    except Exception as e:
        import traceback
        elapsed = time.monotonic() - t0
        log(f"[OCR] 异常: 耗时={elapsed:.2f}s 错误={e}")
        log(f"[OCR] 堆栈:\n{traceback.format_exc()}")
        return _error_response("500", "OCR服务内部异常")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


@app_online.route('/Extract', methods=['POST'])
def api_extract():
    """字段提取接口：按调用方动态 topFields / nerFields 定义返回字段。"""
    t0 = time.monotonic()
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return _error_response("400", "请求体必须是 JSON 对象")

    # V1.1 中三个参数均为必填；两个字段数组允许其中一个为空，但不能同时为空。
    for required_key in ("ocrResult", "topFields", "nerFields"):
        if required_key not in data:
            return _error_response("400", f"缺少 {required_key} 参数")

    ocr_result = data.get("ocrResult")
    top_fields = data.get("topFields")
    ner_fields_def = data.get("nerFields")
    if not isinstance(top_fields, list):
        return _error_response("400", "topFields 必须是数组")
    if not isinstance(ner_fields_def, list):
        return _error_response("400", "nerFields 必须是数组")
    if not top_fields and not ner_fields_def:
        return _error_response("400", "topFields 和 nerFields 不能同时为空")

    class_raw = data.get("class", "")
    title_raw = data.get("title", "")
    if class_raw is not None and not isinstance(class_raw, str):
        return _error_response("400", "class 必须是字符串")
    if title_raw is not None and not isinstance(title_raw, str):
        return _error_response("400", "title 必须是字符串")
    class_input = (class_raw or "").strip()
    title_input = (title_raw or "").strip()

    # 严格校验动态字段定义。
    seen_top_names = set()
    for i, field in enumerate(top_fields):
        if not isinstance(field, dict):
            return _error_response("400", f"topFields[{i}] 必须是对象")
        name = field.get("name")
        field_type = field.get("type")
        description = field.get("description")
        if not isinstance(name, str) or not name.strip():
            return _error_response("400", f"topFields[{i}].name 必须是非空字符串")
        if name in seen_top_names:
            return _error_response("400", f"topFields 存在重复 name: {name}")
        seen_top_names.add(name)
        if field_type not in ("String", "Number", "Boolean"):
            return _error_response("400", f"topFields[{i}].type 必须是 String / Number / Boolean")
        if not isinstance(description, str) or not description.strip():
            return _error_response("400", f"topFields[{i}].description 必须是非空字符串")
        if "required" in field and not isinstance(field.get("required"), bool):
            return _error_response("400", f"topFields[{i}].required 必须是 Boolean")

    seen_ner_names = set()
    for i, field in enumerate(ner_fields_def):
        if not isinstance(field, dict):
            return _error_response("400", f"nerFields[{i}] 必须是对象")
        name = field.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error_response("400", f"nerFields[{i}].name 必须是非空字符串")
        if name in seen_ner_names:
            return _error_response("400", f"nerFields 存在重复 name: {name}")
        seen_ner_names.add(name)
        if "description" in field and field.get("description") is not None and not isinstance(field.get("description"), str):
            return _error_response("400", f"nerFields[{i}].description 必须是字符串")

    try:
        payload = _normalize_ocr_payload(ocr_result)
        pages = payload.get("pages", [])
        markdown_parts = []
        for page in pages:
            md = page.get("markdown", "")
            if md is None:
                md = ""
            if not isinstance(md, str):
                return _error_response("400", "ocrResult.pages[].markdown 必须是字符串")
            if md:
                markdown_parts.append(md)
        markdown = "\n".join(markdown_parts)
        markdown_header, markdown_body = _read_markdown_header(markdown)

        log(
            f"[Extract] 请求: topFields={len(top_fields)}个, "
            f"nerFields={len(ner_fields_def)}个, "
            f"markdown_title={markdown_header['title']}, "
            f"markdown_class={markdown_header['class']}, "
            f"markdown_pagenumber={markdown_header['pagenumber']}"
        )

        # 固定前三行先作为 title/class/pagenumber 读取；NER 仅处理其后的正文。
        ner_result = {}
        ner_completed = False
        if pipeline is not None and getattr(pipeline, "_ner_pipeline", None) is not None and markdown_body:
            ner_result = pipeline._run_ner(markdown_body) or {}
            ner_completed = True
        if not isinstance(ner_result, dict):
            ner_result = {}

        all_aliases = []
        for field in top_fields:
            all_aliases.extend(_field_aliases(field.get("name"), field.get("description")))
        for field in ner_fields_def:
            all_aliases.extend(_field_aliases(field.get("name"), field.get("description", "")))
        all_aliases = list(dict.fromkeys(a for a in all_aliases if a))

        msg = {}
        for field in top_fields:
            name = field["name"]
            desc = field["description"]
            field_type = field["type"]
            required = field.get("required", False)
            aliases = _field_aliases(name, desc)
            kind = _semantic_kind(name, desc)
            value = ""

            # title/class/pagenumber 只读取 Markdown 固定前三行，不再做任何兜底推断。
            if kind:
                value = markdown_header.get(kind, "")
            else:
                value = _match_ner_value(ner_result, aliases)
                if not value:
                    value = _extract_labeled_value(markdown_body, aliases, all_aliases)

            converted = _convert_top_value(value, field_type)
            if converted is None:
                converted = ""  # 类型转换失败视为未识别，不伪造 0/False。
            if converted != "":
                msg[name] = converted
            elif kind or required:
                # 三个固定头字段未识别到时也必须显式返回空字符串。
                msg[name] = ""

        if ner_fields_def:
            ner_msg = {}
            for field in ner_fields_def:
                name = field["name"]
                description = field.get("description", "")
                aliases = _field_aliases(name, description)
                kind = _semantic_kind(name, description)
                if kind:
                    # NER 字段定义中若请求三个固定字段，同样优先读取 Markdown 头。
                    value = markdown_header.get(kind, "")
                    if kind == "pagenumber":
                        value = _normalize_pagenumber(value)
                    ner_msg[name] = value
                    continue
                value = _match_ner_value(ner_result, aliases)
                if not value:
                    value = _extract_labeled_value(markdown_body, aliases, all_aliases)
                # 其他 NER 字段未识别时保持原协议行为：省略。
                if value:
                    ner_msg[name] = value
            msg["NER"] = ner_msg

        # 一张图片实际完成 NER 和字段结果整理后扣减一次额度。
        if ner_completed:
            deal_counter(1)
        elapsed = time.monotonic() - t0
        log(f"[Extract] 成功: 耗时={elapsed:.2f}s")
        return _json_response({"isSuccess": True, "status": "200", "msg": msg})
    except ValueError as e:
        log(f"[Extract] 参数错误: {e}")
        return _error_response("400", str(e))
    except Exception as e:
        import traceback
        elapsed = time.monotonic() - t0
        log(f"[Extract] 异常: 耗时={elapsed:.2f}s 错误={e}")
        log(f"[Extract] 堆栈:\n{traceback.format_exc()}")
        return _error_response("500", "服务内部错误")


# ════════════════════════════════════════════════════════════════
# 启动入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="干部人事档案OCR自动著录服务")
    parser.add_argument("--host", default=cfg.HOST, help="监听地址")
    parser.add_argument("--port-online", type=int, default=cfg.PORT_ONLINE, help="联机版监听端口")
    parser.add_argument("--layout-model", default=cfg.LAYOUT_MODEL_DIR, help="版式分析模型目录")
    parser.add_argument("--cell-model", default=cfg.TABLE_CELL_MODEL_DIR, help="单元格检测模型目录")
    parser.add_argument("--ocr-model", default=cfg.OCR_MODEL_DIR, help="OCR 模型目录")
    parser.add_argument("--ner-model", default=cfg.NER_MODEL_DIR, help="NER 模型目录")
    parser.add_argument("--imgpre-model", default=cfg.IMGPRE_MODEL_DIR, help="图像预处理模型目录")
    parser.add_argument("--yolo-conf", type=float, default=cfg.YOLO_CONF, help="YOLO 置信度阈值")
    parser.add_argument("--gpu", action="store_true", default=None, help="强制使用 GPU")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args, _unknown = parser.parse_known_args()

    global pipeline
    use_gpu = None
    if args.gpu:
        use_gpu = True
    elif args.cpu:
        use_gpu = False

    log("=" * 60)
    log("  干部人事档案OCR自动著录服务")
    log("=" * 60)

    migrate_old_data()
    wait_for_license()
    # 打包环境下 patch paddlex 依赖检查（避免 DependencyError）
    if getattr(sys, '_MEIPASS', None):
        try:
            import paddlex.utils.deps as _deps
            _deps.is_dep_available = lambda dep, **kw: True
            _deps.require_deps = lambda *deps, **kw: None
            _deps.require_extra = lambda *a, **kw: None
            _deps.is_extra_available = lambda extra: True
        except Exception:
            pass

    log("正在加载模型...")
    pipeline = OCRPipeline(use_gpu=use_gpu)
    pipeline.load_models(
        layout_dir=args.layout_model,
        cell_dir=args.cell_model,
        ocr_dir=args.ocr_model,
        ner_dir=args.ner_model,
        imgpre_dir=args.imgpre_model,
        yolo_conf=args.yolo_conf,
    )

    log(f"\n服务启动:")
    log(f"  GPU: {'是' if pipeline.use_gpu else '否'}")
    log(f"  联机版: http://{args.host}:{args.port_online}")
    log(f"    POST /OCR                          — OCR 识别")
    log(f"    POST /Extract                      — 字段提取")
    log(f"  日志文件: {LOG_FILE}")
    log("=" * 60)

    # 启动服务
    app_online.run(host=args.host, port=args.port_online, debug=False, threaded=False)

def wait_for_license():
    while not check_license():
        print(name_validate_license_failed, flush=True)
        print(f'Retrying in {RETRY_INTERVAL}s......', flush=True)
        time.sleep(RETRY_INTERVAL)
    print(name_validate_license_successfully, flush=True)



if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
