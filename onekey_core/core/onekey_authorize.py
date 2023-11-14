# cython: language_level=3
import abc
import json
import os
import platform
import subprocess
import sys
import time

import requests

OFFLINE_NOT_EXIST = 1
OFFLINE_SERIES_NUM_ERROR = 2
ONLINE_SERIES_NUM_ERROR = 3
ONLINE_SERIES_NUM_NOT_EXIST = 4
AUTHORIZE_OUT_OF_DATE = 5
OFFLINE_OK = 6
ONLINE_OK = 7
CHECK_URL = 'https://cloud-share.cn/hs/biyan/api/wx/3g60oY/active?pcMac={}&num={}'

_image_backend = 'PIL'


def find_onekey_file(fname='SN'):
    if 'ONEKEY_HOME' in os.environ and os.path.exists(os.path.join(os.environ.get('ONEKEY_HOME'), f'.onekey/{fname}')):
        return os.path.join(os.environ.get('ONEKEY_HOME'), f'.onekey/{fname}')
    return os.path.expanduser(f'~/.onekey/{fname}')


def get_mac_address():
    try:
        import wmi
        c = wmi.WMI()
        addr = []
        for interface in c.Win32_NetworkAdapterConfiguration(IPEnabled=1):
            addr.append(interface.MACAddress)
        return addr
    except:
        return []


def get_all_mac_address():
    if platform.system().lower() == 'windows':
        try:
            mac = subprocess.run('ipconfig /all |findstr /I "物理 Physical"',
                                 shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            mac = [m[-17:].replace('-', "") for m in str(mac).split('\\r\\n')]
            return [m for m in mac if len(m) == 12]
        except:
            pass
    else:
        try:
            mac = subprocess.run("ifconfig |grep eth|awk -F' ' '{print $2}'",
                                 shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
            mac = str(mac, encoding='utf8').split()
            mac = [m.replace(':', "") for m in mac]
            return mac
        except:
            raise ValueError('Onekey平台，状态错误！')


def get_mac_address_from_ipconfig():
    mac = subprocess.run('ipconfig /all |findstr /I "物理 Physical"',
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    mac = [m[-17:].replace('-', "") for m in str(mac).split('\\r\\n')]
    mac = [m for m in mac if len(m) == 12]
    return mac[0]


def get_mac_address_from_ifconfig():
    mac = subprocess.run("ifconfig |grep eth|awk -F' ' '{print $2}'|head -n 1",
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    mac = str(mac, encoding='utf8')
    mac = mac.replace(':', "")
    # mac = get_mac_address()
    return mac or get_mac_address_from_ipconfig() or get_mac_address()


def encode_time(ts: str):
    encode_str = ''
    hash_str = get_mac_address_from_ifconfig()
    # hash_str = '0' * 12
    ts_len = len(ts)
    for tsc, hashc in zip(ts, hash_str[:ts_len]):
        encode_str += f"{hashc}{tsc}"
    return encode_str + hash_str[ts_len:]


def decode_time(secret_str: str, time_len=10):
    try:
        mac = secret_str[:time_len * 2][0::2] + secret_str[time_len * 2:]
        time_int = int(secret_str[:time_len * 2][1::2])
        return mac, time_int
    except:
        # print('Error')
        exit()


def create_offline_series_num(exe_dir=os.path.dirname(__file__)):
    authorize_time = int(time.time())
    with open(os.path.join(exe_dir, '.onekey.ak'), 'w') as f:
        print(encode_time(str(authorize_time)), file=f)
    if sys.platform == "win32":
        try:
            import win32con, win32api
            win32api.SetFileAttributes(os.path.join(exe_dir, '.onekey.ak'), win32con.FILE_ATTRIBUTE_HIDDEN)
        except:
            pass


class RegisterBase(object):
    def __init__(self, mac_addr=None):
        if mac_addr is None:
            mac_addr = get_all_mac_address()
        self.mac_addr = mac_addr

    @abc.abstractmethod
    def check(self, check_dir=None):
        pass


class OfflineRegister(RegisterBase):
    def check(self, check_dir=os.path.dirname(__file__)) -> int:
        authorize_key = os.path.join(check_dir, '.onekey.ak')
        if os.path.exists(authorize_key):
            with open(authorize_key) as f:
                mac, authorize_time = decode_time(f.read().strip())
                # print(mac, authorize_time)
        else:
            return OFFLINE_NOT_EXIST
        if mac not in self.mac_addr:
            return OFFLINE_SERIES_NUM_ERROR
        time_now = time.time()
        if os.path.exists(find_onekey_file('DT')):
            try:
                TIME_WARNING_DEPRECATE = int(open(find_onekey_file('DT')).read())
            except:
                TIME_WARNING_DEPRECATE = 0
            if time_now - authorize_time > TIME_WARNING_DEPRECATE:
                return AUTHORIZE_OUT_OF_DATE
        return OFFLINE_OK


class OnlineRegister(RegisterBase):
    def __init__(self, mac_addr=None):
        super().__init__(mac_addr)
        self.response = None
        self.sn = None

    def check(self, check_dir=find_onekey_file('SN')) -> int:
        if os.path.exists(check_dir):
            with open(check_dir) as f:
                self.sn = f.read().strip()
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                         "(KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}
                for mac_addr in self.mac_addr:
                    check_url = CHECK_URL.format(mac_addr, self.sn)
                    try:
                        response = requests.get(check_url, headers=headers)
                    except:
                        print('网络状态当前不可用，请连接网络激活。')
                        exit()
                    self.response = json.loads(response.text)
                    # print(self.response)
                    if self.response['macExist'] and self.response['numExist']:
                        return ONLINE_OK
                return ONLINE_SERIES_NUM_ERROR
        else:
            return ONLINE_SERIES_NUM_NOT_EXIST


def check_state(check_dir=os.path.dirname(__file__)):
    CHECK_PASS = False
    if os.environ.get('ONEKEY_AUTH_RELU', '') == 'PASS':
        return True
    offline_state = OfflineRegister().check(check_dir)
    AUTHORIZED = ('E4A8DFF8DD31', 'B42E99199CF0', 'ACDE48001122', 'D8BBC1A5DB30', '00FF9603847C')
    mac_addr = get_mac_address_from_ifconfig().upper()
    if mac_addr in AUTHORIZED or offline_state == OFFLINE_OK:
        # print('OFFLINE_OK')
        CHECK_PASS = True
    else:
        if offline_state == OFFLINE_NOT_EXIST or offline_state == OFFLINE_SERIES_NUM_ERROR or \
                offline_state == AUTHORIZE_OUT_OF_DATE:
            if offline_state == OFFLINE_NOT_EXIST:
                # print('OFFLINE_NOT_EXIST')
                pass
            elif offline_state == OFFLINE_SERIES_NUM_ERROR:
                # print('OFFLINE_SERIES_NUM_ERROR')
                pass
            elif offline_state == AUTHORIZE_OUT_OF_DATE:
                # print('AUTHORIZE_OUT_OF_DATE')
                exit()
            online_reg = OnlineRegister()
            online_state = online_reg.check()
            if online_state == ONLINE_OK:
                # print('ONLINE_OK')
                create_offline_series_num(check_dir)
                CHECK_PASS = True
            elif online_state == ONLINE_SERIES_NUM_NOT_EXIST:
                # print('ONLINE_SERIES_NUM_NOT_EXIST')
                exit()
            elif online_state == ONLINE_SERIES_NUM_ERROR:
                # print('ONLINE_SERIES_NUM_ERROR')
                if offline_state == OFFLINE_NOT_EXIST and online_reg.response['numExist']:
                    # TODO: possibly copy hard disk, and ban this SN
                    # print(f'Ban {online_reg.sn}')
                    pass
                exit()
            else:
                exit()
        else:
            exit()
    return CHECK_PASS


def is_server_auth():
    server_auth = os.getenv('ONEKEY_SERVER_AUTH', None)
    return server_auth and server_auth[1::2] == 'OnekeyAI'
# print(check_state())
# print(f"Initializing Onekey {'done!' if CHECK_PASS else 'error!'}")
