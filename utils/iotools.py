# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from PIL import Image, ImageFile
import errno
import json
import pickle as pkl
import os
import os.path as osp
import yaml
from easydict import EasyDict as edict
from typing import Any, Dict, List
import os.path as op

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Read one image, reporting damaged inputs instead of retrying forever."""
    if not osp.exists(img_path):
        raise FileNotFoundError(f"Image does not exist: {img_path}")
    try:
        with Image.open(img_path) as image:
            return image.convert('RGB')
    except (OSError, ValueError) as error:
        raise OSError(f"Unable to decode image: {img_path}") from error


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def read_json_from_LLM_captions(file_path: str) -> Dict[str, Any] | List[Any]:
    """
    读取JSON文件并返回解析后的字典或列表，兼容多种编码
    """
    if not op.exists(file_path):
        raise FileNotFoundError(f"JSON文件不存在: {file_path}")

    # 尝试多种编码方式读取，解决编码不匹配问题
    encodings = ['utf-8', 'latin-1', 'gbk']  # 优先utf-8，其次兼容编码
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue  # 尝试下一种编码
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON文件解析错误（{file_path}）: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"读取JSON文件失败（{file_path}）: {str(e)}")

    # 如果所有编码都尝试失败，抛出明确错误
    raise RuntimeError(f"所有编码方式均无法解析文件 {file_path}，请检查文件编码格式")



def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def get_text_embedding(path, length):
    with open(path, 'rb') as f:
        word_frequency = pkl.load(f)


def save_train_configs(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/configs.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

def load_train_configs(path):
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)
