import json
import pathlib
import re
import subprocess
import sys
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml


ROOT = pathlib.Path(__file__).resolve().parent
WEBUI_PAGE = ROOT / 'webui.html'
EXPERIMENTS_DIR = ROOT / 'experiments'
MAX_REQUEST_SIZE = 64 * 1024
EXPERIMENT_NAME_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')

state_lock = threading.Lock()
stop_event = threading.Event()
active_process = None
state = {
    'status': 'idle',
    'stage': '',
    'message': '等待开始',
    'logs': deque(maxlen=500),
}


def add_log(message):
    with state_lock:
        state['logs'].append(message.rstrip())


def set_status(status, stage='', message=''):
    with state_lock:
        state.update(status=status, stage=stage, message=message)


def get_state():
    with state_lock:
        return {
            'status': state['status'],
            'stage': state['stage'],
            'message': state['message'],
            'logs': list(state['logs']),
        }


def run_command(command, stage):
    global active_process

    set_status('running', stage, '正在执行')
    add_log(f'> {" ".join(str(part) for part in command)}')
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0),
    )
    with state_lock:
        active_process = process

    try:
        if stop_event.is_set():
            process.terminate()
        if process.stdout is not None:
            for line in process.stdout:
                add_log(line)
                if stop_event.is_set() and process.poll() is None:
                    process.terminate()
        return_code = process.wait()
    finally:
        with state_lock:
            active_process = None

    if stop_event.is_set():
        raise InterruptedError('用户已停止任务')
    if return_code != 0:
        raise RuntimeError(f'{stage}进程退出，返回码：{return_code}')


def run_finetune(options):
    global active_process

    set_status('running', 'preprocessing', '正在预处理音频')
    try:
        experiment_dir = EXPERIMENTS_DIR / options['experiment_name']
        data_dir = experiment_dir / 'data'
        processed_dir = experiment_dir / 'preprocessed'
        config_path = experiment_dir / 'webui_config.yaml'
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config = {
            'base_config': ['configs/ft_hifigan.yaml'],
            'data_input_path': [str(options['audio_dir'])],
            'data_out_path': [str(processed_dir)],
            'val_num': options['val_num'],
            'DataIndexPath': str(data_dir),
            'finetune_enabled': True,
            'finetune_ckpt_path': str(options['checkpoint']),
            'max_updates': options['max_steps'],
            'batch_size': options['batch_size'],
            'ds_workers': options['preprocess_workers'],
            'freezing_enabled': options['freeze_mpd'],
            'frozen_params': ['discriminator.mpd'] if options['freeze_mpd'] else [],
        }
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding='utf-8',
        )

        run_command(
            [
                sys.executable, 'process.py',
                '--config', str(config_path),
                '--num_cpu', str(options['preprocess_workers']),
                '--strx', '1',
            ],
            'preprocessing',
        )
        if stop_event.is_set():
            raise InterruptedError('用户已停止任务')

        run_command(
            [
                sys.executable, 'train.py',
                '--config', str(config_path),
                '--exp_name', options['experiment_name'],
                '--work_dir', str(EXPERIMENTS_DIR),
            ],
            'training',
        )
        set_status('completed', 'completed', '微调完成')
        add_log('微调已完成。')
    except InterruptedError as error:
        set_status('stopped', 'stopped', str(error))
        add_log(str(error))
    except Exception as error:
        set_status('failed', state.get('stage', ''), str(error))
        add_log(f'错误：{error}')
    finally:
        with state_lock:
            active_process = None


def run_export(options):
    set_status('running', 'exporting', '正在导出模型和配置')
    try:
        options['output_dir'].mkdir(parents=True, exist_ok=True)
        model_path = options['output_dir'] / 'model.ckpt'
        config_path = options['output_dir'] / 'config.json'
        run_command(
            [
                sys.executable, 'export_ckpt.py',
                '--ckpt_path', str(options['checkpoint']),
                '--save_path', str(model_path),
            ],
            'exporting',
        )
        if not model_path.is_file() or not config_path.is_file():
            raise RuntimeError('导出命令结束，但没有生成 model.ckpt 和 config.json。')
        set_status('completed', 'completed', '模型和配置导出完成')
        add_log(f'模型：{model_path}')
        add_log(f'配置：{config_path}')
    except InterruptedError as error:
        set_status('stopped', 'stopped', str(error))
        add_log(str(error))
    except Exception as error:
        set_status('failed', 'exporting', str(error))
        add_log(f'错误：{error}')
    finally:
        with state_lock:
            active_process = None


def validate_options(payload):
    if not isinstance(payload, dict):
        raise ValueError('请求内容格式无效。')

    experiment_name = str(payload.get('experiment_name', '')).strip()
    if not EXPERIMENT_NAME_PATTERN.fullmatch(experiment_name):
        raise ValueError('实验名称需为 1-64 位字母、数字、点、下划线或连字符，且以字母或数字开头。')

    audio_text = str(payload.get('audio_dir', '')).strip()
    checkpoint_text = str(payload.get('checkpoint', '')).strip()
    if not audio_text or not checkpoint_text:
        raise ValueError('请填写音频目录和预训练权重路径。')

    audio_dir = pathlib.Path(audio_text).expanduser().resolve(strict=True)
    checkpoint = pathlib.Path(checkpoint_text).expanduser().resolve(strict=True)
    if not audio_dir.is_dir():
        raise ValueError('音频路径必须是目录。')
    if not checkpoint.is_file():
        raise ValueError('预训练权重路径必须是文件。')
    if checkpoint.suffix.lower() != '.ckpt':
        raise ValueError('预训练权重应为 .ckpt 文件。')

    def bounded_integer(field, label, default, minimum, maximum):
        try:
            value = int(payload.get(field, default))
        except (TypeError, ValueError):
            raise ValueError(f'{label}必须是整数。') from None
        if not minimum <= value <= maximum:
            raise ValueError(f'{label}范围为 {minimum} 到 {maximum}。')
        return value

    freeze_mpd = payload.get('freeze_mpd', False)
    if not isinstance(freeze_mpd, bool):
        raise ValueError('冻结 MPD 参数无效。')

    return {
        'experiment_name': experiment_name,
        'audio_dir': audio_dir,
        'checkpoint': checkpoint,
        'max_steps': bounded_integer('max_steps', '训练步数', 2000, 1, 10_000_000),
        'batch_size': bounded_integer('batch_size', 'Batch size', 4, 1, 256),
        'val_num': bounded_integer('val_num', '验证音频数', 5, 1, 10_000),
        'preprocess_workers': bounded_integer(
            'preprocess_workers', '预处理线程数', 4, 1, 64
        ),
        'freeze_mpd': freeze_mpd,
    }


def validate_export_options(payload):
    if not isinstance(payload, dict):
        raise ValueError('请求内容格式无效。')
    checkpoint_text = str(payload.get('export_checkpoint', '')).strip()
    output_text = str(payload.get('export_dir', '')).strip()
    if not checkpoint_text or not output_text:
        raise ValueError('请填写待导出的 checkpoint 和目标文件夹。')

    checkpoint = pathlib.Path(checkpoint_text).expanduser().resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.suffix.lower() != '.ckpt':
        raise ValueError('待导出模型必须是 .ckpt 文件。')
    config_path = checkpoint.with_name('config.yaml')
    if not config_path.is_file():
        raise ValueError(f'找不到 checkpoint 对应的配置文件：{config_path}')

    output_dir = pathlib.Path(output_text).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError('导出目标必须是文件夹。')
    if (output_dir / 'model.ckpt').exists() or (output_dir / 'config.json').exists():
        raise ValueError('目标文件夹中已存在 model.ckpt 或 config.json，请选择其他文件夹。')
    return {'checkpoint': checkpoint, 'output_dir': output_dir}


class WebUIHandler(BaseHTTPRequestHandler):
    def send_json(self, status_code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/':
            try:
                page = WEBUI_PAGE.read_bytes()
            except OSError as error:
                self.send_json(500, {'error': f'无法读取 Web UI 页面：{error}'})
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif self.path == '/api/status':
            self.send_json(200, get_state())
        else:
            self.send_json(404, {'error': '找不到请求的页面。'})

    def do_POST(self):
        global active_process

        if self.path not in ('/api/start', '/api/stop', '/api/browse', '/api/export'):
            self.send_json(404, {'error': '找不到请求的接口。'})
            return

        if self.path == '/api/browse':
            try:
                content_length = int(self.headers.get('Content-Length', '0'))
                if content_length < 0 or content_length > MAX_REQUEST_SIZE:
                    self.send_json(413, {'error': '请求内容过大。'})
                    return
                payload = json.loads(self.rfile.read(content_length).decode('utf-8'))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                self.send_json(400, {'error': '请求内容不是有效的 JSON。'})
                return

            if not isinstance(payload, dict) or payload.get('kind') not in (
                'directory', 'checkpoint', 'export_directory'
            ):
                self.send_json(400, {'error': '请选择有效的路径类型。'})
                return

            try:
                import tkinter
                from tkinter import filedialog

                root = tkinter.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                try:
                    if payload['kind'] in ('directory', 'export_directory'):
                        selected_path = filedialog.askdirectory(
                            title=(
                                '选择导出目标文件夹'
                                if payload['kind'] == 'export_directory'
                                else '选择训练音频目录'
                            ),
                            initialdir=str(ROOT),
                            parent=root,
                        )
                    else:
                        selected_path = filedialog.askopenfilename(
                            title='选择预训练权重',
                            initialdir=str(ROOT),
                            filetypes=[('Checkpoint 文件', '*.ckpt'), ('所有文件', '*.*')],
                            parent=root,
                        )
                finally:
                    root.destroy()
            except ImportError as error:
                self.send_json(500, {'error': f'当前 Python 环境不支持系统路径选择框：{error}'})
                return
            except tkinter.TclError as error:
                self.send_json(500, {'error': f'无法打开系统路径选择框：{error}'})
                return
            self.send_json(200, {'path': selected_path})
            return

        try:
            content_length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.send_json(400, {'error': '请求长度无效。'})
            return
        if content_length < 0 or content_length > MAX_REQUEST_SIZE:
            self.send_json(413, {'error': '请求内容过大。'})
            return

        if self.path == '/api/stop':
            with state_lock:
                if state['status'] != 'running':
                    self.send_json(409, {'error': '当前没有正在运行的任务。'})
                    return
                stop_event.set()
                process = active_process
                state.update(status='stopping', message='正在停止任务')
            if process is not None and process.poll() is None:
                process.terminate()
            self.send_json(200, get_state())
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {'error': '请求内容不是有效的 JSON。'})
            return

        try:
            if self.path == '/api/export':
                options = validate_export_options(payload)
                target = run_export
            else:
                options = validate_options(payload)
                target = run_finetune
        except (OSError, ValueError) as error:
            self.send_json(400, {'error': str(error)})
            return

        with state_lock:
            if state['status'] in ('running', 'stopping'):
                self.send_json(409, {'error': '已有训练或导出任务正在运行。'})
                return
            stop_event.clear()
            state['logs'].clear()
            state.update(
                status='running',
                stage='exporting' if self.path == '/api/export' else 'preprocessing',
                message='正在导出模型和配置' if self.path == '/api/export' else '正在预处理音频',
            )
        thread = threading.Thread(target=target, args=(options,), daemon=True)
        thread.start()
        self.send_json(202, get_state())

    def log_message(self, format_string, *args):
        return


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SingingVocoders 本地微调 Web UI')
    parser.add_argument('--port', type=int, default=7860, help='本地 Web UI 端口')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('端口必须在 1 到 65535 之间。')

    server = ThreadingHTTPServer(('127.0.0.1', args.port), WebUIHandler)
    print(f'微调 Web UI 已启动：http://127.0.0.1:{args.port}')
    print('仅监听本机地址。按 Ctrl+C 退出。')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n正在关闭 Web UI。')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
